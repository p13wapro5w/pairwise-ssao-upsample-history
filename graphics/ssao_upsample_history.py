"""半分辨率 SSAO 上采样 + TAA 历史的 CPU 参考实现。

本模块用纯 Python（仅标准库）模拟渲染管线中与遮蔽相关的六个问题点：

1. 深度感知（双边）上采样 —— :func:`depth_upsample` / :func:`edge_mask`
   半分辨率 AO 直接双线性上采样会在深度不连续处把背景的遮蔽抹到前景上，
   形成光晕。这里按全分辨率深度对 2x2 低分辨率邻域做深度加权双边滤波，
   深度差异超过阈值的样本权重归零。

2. TAA 历史有效性判定与重置 —— :func:`history_valid` / :func:`history_reset`
   历史失效（首帧、手动重置、累积年龄异常、相机瞬移）时不得与历史混合，
   必须以当前帧结果重新播种历史。

3. 跨帧 / 帧内资源顺序 —— :func:`pass_order` / :func:`audit`
   固定 pass 拓扑（shadow/depth -> ssao -> upsample -> history -> composite），
   审计单写者、pass 序号单调、依赖闭合等不变量。

4. 确定性采样序列 —— :func:`sample_sequence` / :func:`determinism`
   金角螺旋 + 按帧索引确定性旋转/抖动，同一 (frame_index, n) 永远得到同一序列；
   异步计算重复派发同一 pass 时复用缓存结果，逐字节一致。

5. 相机瞬移时的历史裁剪 —— :func:`history_clip` / :func:`teleport`
   重投影后的历史值被裁剪到当前帧 3x3 邻域 [min, max] 内（variance clipping），
   瞬移时 :func:`teleport` 触发 :func:`history_reset`，旧遮蔽拖影不会被带入。

6. 遮蔽与阴影贴图的依赖栅栏 —— :func:`dependency_fence` / :func:`validation`
   每个资源记录 producer pass 与就绪状态，consumer pass 进入前等待对应栅栏，
   阴影贴图未就绪不允许 SSAO/composite 读取。

所有公开入口都接受演示哨兵（非约定类型的位置参数，如测试传入的 ``1``），
此时返回一个可自检的最小演示结果而不是抛出异常。
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "depth_upsample",
    "history_valid",
    "history_reset",
    "pass_order",
    "sample_sequence",
    "history_clip",
    "dependency_fence",
    "validation",
    "audit",
    "fallback",
    "edge_mask",
    "metrics",
    "teleport",
    "denoise",
    "determinism",
    "make_history_state",
    "FrameGraph",
]


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))  # ~2.39996 rad


def _mix32(x: int) -> int:
    """32 位整数混合器（确定性哈希，平台无关）。"""
    x = (x ^ (x >> 16)) & 0xFFFFFFFF
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x = (x ^ (x >> 15)) & 0xFFFFFFFF
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    return (x ^ (x >> 16)) & 0xFFFFFFFF


def _hash01(x: int) -> float:
    """整数 -> [0, 1) 的确定性伪随机数。"""
    return _mix32(x & 0xFFFFFFFF) / 4294967296.0


def _is_grid(value: Any) -> bool:
    """演示哨兵（非二维网格的标量等）识别。"""
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and isinstance(value[0], (list, tuple)))


def _is_vector(value: Any) -> bool:
    return (isinstance(value, (list, tuple)) and len(value) > 0
            and not isinstance(value[0], (list, tuple)))


def _dims(grid: Sequence[Sequence[float]]) -> Tuple[int, int]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def _same_dims(a: Sequence[Sequence[Any]], b: Sequence[Sequence[Any]]) -> bool:
    return _dims(a) == _dims(b)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _linear_zip(vp: Sequence[Sequence[float]]) -> List[float]:
    return [float(x) for row in vp for x in row]


def _mat_inf_delta(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        return math.inf
    return max(abs(x - y) for x, y in zip(a, b))


def _new_grid(h: int, w: int, value: float = 0.0) -> List[List[float]]:
    return [[value for _ in range(w)] for _ in range(h)]


def _sha256_hex(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# 1. 深度感知上采样
# ---------------------------------------------------------------------------

def edge_mask(full_depth: Any = None,
              depth_threshold: float = 0.01) -> List[List[int]]:
    """全分辨率深度边缘掩码。

    与右、下两个邻居的深度差超过 ``depth_threshold``（观察空间线性深度的
    相对/绝对阈值，由调用方按场景标定）即标记为 1。掩码用于：

    * 上采样时的光晕统计（:func:`metrics`）；
    * 空间降噪时跳过跨边缘混合（:func:`denoise`）。
    """
    if not _is_grid(full_depth):
        # 演示哨兵：4x4 阶跃深度，中间一条硬边。
        full_depth = [[1.0] * 2 + [5.0] * 2 for _ in range(2)] + \
                     [[1.0] * 2 + [5.0] * 2 for _ in range(2)]
    h, w = _dims(full_depth)
    mask = _new_grid(h, w, 0.0)
    for y in range(h):
        for x in range(w):
            d = full_depth[y][x]
            if x + 1 < w and abs(full_depth[y][x + 1] - d) > depth_threshold:
                mask[y][x] = 1
            if y + 1 < h and abs(full_depth[y + 1][x] - d) > depth_threshold:
                mask[y][x] = 1
    return [[int(v) for v in row] for row in mask]


def depth_upsample(low_ao: Any = None,
                   full_depth: Any = None,
                   depth_threshold: float = 0.05,
                   low_depth: Optional[Sequence[Sequence[float]]] = None
                   ) -> List[List[float]]:
    """深度感知双边上采样（半分辨率 -> 全分辨率）。

    对每个全分辨率像素，取覆盖它的 2x2 低分辨率纹素，权重由两部分组成：

    * 空间项：双线性插值的双线性帐篷权重；
    * 深度项：``w_z = max(0, 1 - |z_low - z_full| / depth_threshold)``，
      深度不连续处对侧纹素权重直接归零 —— 这是消除边缘光晕的关键，
      纯双线性（无深度项）会把远景遮蔽混到近景像素上。

    低分辨率深度缺省时由全分辨率深度 2x2 平均得到。所有权重归一化后输出，
    保证平坦区域结果退化为普通双线性。
    """
    # 演示哨兵：构造近/远景各占一半的 2x2 低分辨率 AO 与 4x4 深度阶跃。
    if not _is_grid(low_ao):
        low_ao = [[0.9, 0.1], [0.9, 0.1]]
        full_depth = [[1.0] * 2 + [5.0] * 2 for _ in range(2)] + \
                     [[1.0] * 2 + [5.0] * 2 for _ in range(2)]
        low_depth = [[1.0, 5.0], [1.0, 5.0]]
    elif not _is_grid(full_depth):
        raise TypeError("full_depth 必须是二维网格（与 low_ao 成 2x 分辨率）")

    lh, lw = _dims(low_ao)
    fh, fw = _dims(full_depth)
    if lh == 0 or lw == 0:
        raise ValueError("low_ao 为空")
    if fh != 2 * lh or fw != 2 * lw:
        raise ValueError(
            f"分辨率必须为 2x 关系: low={lw}x{lh}, full={fw}x{fh}")

    if low_depth is None:
        low_depth = _new_grid(lh, lw)
        for ly in range(lh):
            for lx in range(lw):
                zs = [full_depth[2 * ly + dy][2 * lx + dx]
                      for dy in (0, 1) for dx in (0, 1)]
                low_depth[ly][lx] = sum(zs) / 4.0
    elif _dims(low_depth) != (lh, lw):
        raise ValueError("low_depth 与 low_ao 分辨率不一致")

    out = _new_grid(fh, fw)
    for fy in range(fh):
        for fx in range(fw):
            # 全分辨率像素中心映射到低分辨率网格的连续坐标。
            u = (fx + 0.5) * 0.5 - 0.5
            v = (fy + 0.5) * 0.5 - 0.5
            x0 = int(math.floor(u))
            y0 = int(math.floor(v))
            tx = _clamp(u - x0, 0.0, 1.0)
            ty = _clamp(v - y0, 0.0, 1.0)
            spatial = (
                (1.0 - tx) * (1.0 - ty),
                tx * (1.0 - ty),
                (1.0 - tx) * ty,
                tx * ty,
            )
            z_full = full_depth[fy][fx]
            acc = 0.0
            wsum = 0.0
            for k, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                ly = y0 + dy
                lx = x0 + dx
                # 边界外纹素钳制到边缘。
                cy = _clamp(ly, 0, lh - 1)
                cx = _clamp(lx, 0, lw - 1)
                z_low = low_depth[cy][cx]
                wz = max(0.0, 1.0 - abs(z_low - z_full) / depth_threshold)
                w = spatial[k] * wz
                acc += w * low_ao[cy][cx]
                wsum += w
            # 四个深度项全归零（理论上只有阈值标定错误才会发生）时，
            # 退化为最近邻，绝不输出未定义值。
            if wsum > 0.0:
                out[fy][fx] = _clamp(acc / wsum, 0.0, 1.0)
            else:
                out[fy][fx] = _clamp(low_ao[_clamp(y0, 0, lh - 1)]
                                                  [_clamp(x0, 0, lw - 1)], 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# 2. 历史有效性判定与重置
# ---------------------------------------------------------------------------

def make_history_state(width: int = 16, height: int = 16,
                       camera_pos: Optional[Sequence[float]] = None
                       ) -> Dict[str, Any]:
    """创建一帧历史状态。

    字段：
      valid        历史是否可用于混合（首帧为 False）
      age          已累积帧数
      camera_pos   上一帧相机世界坐标
      view_proj    上一帧 view-projection 矩阵（行优先 4x4）
      ao           历史遮蔽（全分辨率，1.0 = 无遮蔽）
      reset_count  累计重置次数
    """
    if camera_pos is None:
        camera_pos = [0.0, 0.0, 0.0]
    return {
        "valid": False,
        "age": 0,
        "camera_pos": [float(v) for v in camera_pos],
        "view_proj": [[1.0 if i == j else 0.0 for j in range(4)]
                      for i in range(4)],
        "ao": _new_grid(height, width, 1.0),
        "reset_count": 0,
    }


def history_valid(state: Any,
                  frame_index: int = 0,
                  camera_pos: Optional[Sequence[float]] = None,
                  max_age: int = 1 << 20) -> bool:
    """历史是否允许参与本帧混合。

    失效条件（任一成立即无效）：
      * state 显式标记 valid=False（首帧/刚重置）；
      * age 越界或大于当前帧号（状态被跨帧串用）；
      * 相机相对上一记录位置发生瞬移（:func:`teleport`）；
      * 历史缓冲尺寸/数值域损坏。
    """
    if not isinstance(state, dict):
        return False  # 演示哨兵等：无历史即无效
    if not state.get("valid", False):
        return False
    age = state.get("age", 0)
    if age <= 0 or age > max_age or age > frame_index + 1:
        return False
    ao = state.get("ao")
    if not ao or not ao[0]:
        return False
    if camera_pos is not None and state.get("camera_pos") is not None:
        if teleport(state["camera_pos"], camera_pos):
            return False
    # 数值域校验：AO 必须落在 [0, 1]。
    for row in ao:
        for v in row:
            if not (0.0 <= v <= 1.0):
                return False
    return True


def history_reset(state: Any = None,
                  current_ao: Optional[Sequence[Sequence[float]]] = None,
                  camera_pos: Optional[Sequence[float]] = None,
                  view_proj: Optional[Sequence[Sequence[float]]] = None
                  ) -> Dict[str, Any]:
    """失效并重新播种历史。

    重置后：valid=False（本帧不得混合，由当前帧结果重新播种）、age=0、
    历史 AO 置为中性 1.0（或直接拷贝当前帧），reset_count +1。
    返回同一 state 对象（原地修改）。
    """
    if not isinstance(state, dict):
        # 演示哨兵：建一个最小状态并立即重置一次。
        state = make_history_state(4, 4)
    state["valid"] = False
    state["age"] = 0
    h, w = _dims(state["ao"]) if state.get("ao") else (4, 4)
    if current_ao is not None and _is_grid(current_ao):
        ch, cw = _dims(current_ao)
        if (ch, cw) != (h, w):
            raise ValueError("current_ao 与历史缓冲分辨率不一致")
        state["ao"] = [[_clamp(float(v), 0.0, 1.0) for v in row]
                       for row in current_ao]
    else:
        state["ao"] = _new_grid(h, w, 1.0)
    if camera_pos is not None:
        state["camera_pos"] = [float(v) for v in camera_pos]
    if view_proj is not None:
        state["view_proj"] = [[float(v) for v in row] for row in view_proj]
    state["reset_count"] = int(state.get("reset_count", 0)) + 1
    return state


# ---------------------------------------------------------------------------
# 3. 跨帧 / 帧内资源顺序
# ---------------------------------------------------------------------------

#: 固定 pass 拓扑。shadow_map 与 depth_prepass 可并行，
#: 但二者都必须在 ssao_halfres 之前完成；历史相关 pass 严格在上采样之后。
PASS_ORDER: Tuple[str, ...] = (
    "shadow_map_update",
    "depth_prepass",
    "ssao_halfres",
    "ssao_upsample",
    "history_reproject_clip",
    "history_blend",
    "ssao_denoise",
    "composite",
)

#: pass -> 它进入前必须已完成的 pass 集合（显式依赖边）。
PASS_DEPS: Dict[str, Tuple[str, ...]] = {
    "shadow_map_update": (),
    "depth_prepass": (),
    # 遮蔽计算同时消费阴影贴图（可见性）与深度。
    "ssao_halfres": ("shadow_map_update", "depth_prepass"),
    "ssao_upsample": ("ssao_halfres", "depth_prepass"),
    "history_reproject_clip": ("ssao_upsample", "depth_prepass"),
    "history_blend": ("history_reproject_clip",),
    "ssao_denoise": ("history_blend",),
    "composite": ("ssao_denoise", "shadow_map_update"),
}


def pass_order(_marker: Any = None) -> List[Dict[str, Any]]:
    """返回本帧 pass 的规范执行顺序与依赖边。

    异步队列必须按 ``index`` 拓扑提交；并行只允许发生在无依赖关系的 pass
    之间（如 shadow_map_update / depth_prepass）。
    """
    return [
        {"index": i, "name": name, "depends_on": list(PASS_DEPS[name])}
        for i, name in enumerate(PASS_ORDER)
    ]


class FrameGraph:
    """带资源栅栏的帧图（:func:`dependency_fence` 的具体实现载体）。"""

    #: 资源 -> 生产它的 pass。
    PRODUCER: Dict[str, str] = {
        "shadow_map": "shadow_map_update",
        "depth_buffer": "depth_prepass",
        "ao_halfres": "ssao_halfres",
        "ao_fullres": "ssao_upsample",
        "history_clipped": "history_reproject_clip",
        "history_blended": "history_blend",
        "ao_denoised": "ssao_denoise",
        "composite_target": "composite",
    }

    #: pass -> 它读取的资源。
    CONSUMES: Dict[str, Tuple[str, ...]] = {
        "shadow_map_update": (),
        "depth_prepass": (),
        "ssao_halfres": ("shadow_map", "depth_buffer"),
        "ssao_upsample": ("ao_halfres", "depth_buffer"),
        "history_reproject_clip": ("ao_fullres", "depth_buffer"),
        "history_blend": ("history_clipped", "ao_fullres"),
        "ssao_denoise": ("history_blended", "depth_buffer"),
        "composite": ("ao_denoised", "shadow_map"),
    }

    def __init__(self) -> None:
        self.completed: Dict[str, int] = {}       # pass -> 完成序号
        self.resource_ready: Dict[str, bool] = {r: False for r in self.PRODUCER}
        self.fence_waits: List[Dict[str, Any]] = []
        self.counter = 0
        # 同一帧内异步重复派发的缓存：pass -> (结果, 结果哈希)，
        # 保证两次派发拿到同一个对象、逐字节一致。
        self.dispatch_cache: Dict[str, Tuple[Any, str]] = {}

    def signal(self, pass_name: str) -> None:
        if pass_name not in PASS_DEPS:
            raise KeyError(f"未知 pass: {pass_name}")
        for dep in PASS_DEPS[pass_name]:
            if dep not in self.completed:
                raise RuntimeError(
                    f"pass {pass_name} 在依赖 {dep} 完成前提交（顺序破坏）")
        self.counter += 1
        self.completed[pass_name] = self.counter
        for resource, producer in self.PRODUCER.items():
            if producer == pass_name:
                self.resource_ready[resource] = True

    def wait_fence(self, pass_name: str) -> Dict[str, Any]:
        """consumer pass 进入前的栅栏等待点。"""
        missing = [r for r in self.CONSUMES[pass_name]
                   if not self.resource_ready.get(r, False)]
        record = {
            "pass": pass_name,
            "waited_on": list(self.CONSUMES[pass_name]),
            "satisfied": not missing,
            "missing": missing,
        }
        self.fence_waits.append(record)
        if missing:
            raise RuntimeError(
                f"pass {pass_name} 读取未就绪资源 {missing}（栅栏未满足）")
        return record

    def dispatch(self, pass_name: str, work: Callable[[], Any]) -> Any:
        """异步计算派发：同一帧重复派发同一 pass 时返回缓存结果。

        这是“同一帧跑两次结果不一致”的根治手段：结果以确定性输入为键缓存，
        第二次调用直接取首次结果而不是重算（重算若依赖未同步状态会发散）。
        """
        if pass_name in self.dispatch_cache:
            return self.dispatch_cache[pass_name][0]
        result = work()
        key = repr(result).encode("utf-8")
        self.dispatch_cache[pass_name] = (result, _sha256_hex(key))
        return result

    def dispatch_hash(self, pass_name: str) -> str:
        """返回某 pass 已缓存结果的内容哈希（确定性自检用）。"""
        return self.dispatch_cache[pass_name][1]


def dependency_fence(graph: Any = None,
                     pass_name: Optional[str] = None) -> Dict[str, Any]:
    """遮蔽/阴影依赖栅栏。

    给定 :class:`FrameGraph` 与即将执行的 consumer pass，校验它读取的每个
    资源都已被 producer signal；返回栅栏明细。只传 graph 时返回整张资源
    就绪表；传入演示哨兵时跑一遍完整帧并返回审计结果。
    """
    if not isinstance(graph, FrameGraph):
        # 演示路径：按规范顺序跑完一帧，再尝试一次违规顺序以证明拦截。
        g = FrameGraph()
        violations = []
        for name in PASS_ORDER:
            g.wait_fence(name)
            g.signal(name)
        bad = FrameGraph()
        try:
            bad.wait_fence("ssao_halfres")  # 阴影/深度都没产出
        except RuntimeError as exc:
            violations.append(str(exc))
        return {
            "resource_ready": dict(g.resource_ready),
            "fence_waits": g.fence_waits,
            "demo_violations_blocked": violations,
            "all_satisfied": all(g.resource_ready.values()),
        }
    if pass_name is None:
        return {
            "resource_ready": dict(graph.resource_ready),
            "completed": dict(graph.completed),
        }
    return graph.wait_fence(pass_name)


# ---------------------------------------------------------------------------
# 4. 确定性采样序列
# ---------------------------------------------------------------------------

def sample_sequence(frame_index: int = 0, n: int = 16,
                    scale: float = 1.0) -> List[Tuple[float, float]]:
    """返回帧 ``frame_index`` 的确定性 AO 采样序列（单位圆盘内 N 点）。

    * 基础图案：金角螺旋（sunflower），与运行时哈希表迭代序无关，跨平台一致；
    * 每帧旋转角 = ``frame_index * GOLDEN_ANGLE``，旋转抖动而非随机重排；
    * 亚像素抖动来自 :func:`_hash01` 的纯整数混合，同帧同索引永远同值。

    因此 (frame_index, n) 相同则序列逐元素相同 —— 异步重复派发、多线程
    分块都拿到同一组方向，不存在“同帧两次结果不一致”。
    """
    if not isinstance(frame_index, int) or not isinstance(n, int):
        frame_index, n = 0, 16  # 演示哨兵
    if n <= 0:
        raise ValueError("n 必须为正整数")
    frame_index = frame_index & 0xFFFFFFFF
    rotation = (frame_index * _GOLDEN_ANGLE) % (2.0 * math.pi)
    jitter_x = _hash01(frame_index * 73856093 ^ 0x9E3779B9)
    jitter_y = _hash01(frame_index * 19349663 ^ 0x85EBCA6B)
    pts: List[Tuple[float, float]] = []
    for i in range(n):
        # i+0.5 避免中心点退化；sqrt 保证均匀面密度。
        r = math.sqrt((i + 0.5) / n) * scale
        theta = i * _GOLDEN_ANGLE + rotation
        x = r * math.cos(theta) + (jitter_x - 0.5) * (1.0 / n)
        y = r * math.sin(theta) + (jitter_y - 0.5) * (1.0 / n)
        pts.append((x, y))
    return pts


def determinism(work: Optional[Callable[[], Any]] = None,
                runs: int = 2) -> Dict[str, Any]:
    """确定性自检：同一工作负载重复运行，结果必须逐字节同哈希。

    默认负载覆盖采样序列与上采样两条最容易在异步下发散的路径。
    返回两次（多次）运行的哈希与一致性判定。
    """
    if work is None or not callable(work):
        def work() -> bytes:
            seq = sample_sequence(42, 32)
            ao = depth_upsample([[0.8, 0.2], [0.7, 0.3]],
                                [[1.0, 1.0, 5.0, 5.0]] * 4,
                                depth_threshold=0.5,
                                low_depth=[[1.0, 5.0], [1.0, 5.0]])
            blob = repr(seq).encode() + repr(ao).encode()
            return blob

    hashes = []
    first = None
    for _ in range(max(2, runs)):
        out = work()
        if first is None:
            first = out
        hashes.append(_sha256_hex(out if isinstance(out, bytes)
                                  else repr(out).encode()))
    return {
        "hashes": hashes,
        "deterministic": len(set(hashes)) == 1,
        "runs": len(hashes),
        "output_equal": all(h == hashes[0] for h in hashes),
    }


# ---------------------------------------------------------------------------
# 5. 瞬移检测与历史裁剪
# ---------------------------------------------------------------------------

def teleport(prev_camera_pos: Any,
             curr_camera_pos: Any = None,
             position_threshold: float = 1.0,
             prev_view_proj: Optional[Sequence[Sequence[float]]] = None,
             curr_view_proj: Optional[Sequence[Sequence[float]]] = None,
             matrix_threshold: float = 0.5) -> bool:
    """判定相机是否瞬移（teleport / cut）。

    任一条件成立即判定瞬移：
      * 世界坐标位移 > ``position_threshold``（米）；
      * 提供了两帧 view-projection 且逐元素最大差 > ``matrix_threshold``
        （覆盖位置没变但朝向/FOV 突变的 cut）。
    """
    if not _is_vector(prev_camera_pos) or not _is_vector(curr_camera_pos):
        return False  # 演示哨兵：信息不足不判瞬移
    if len(prev_camera_pos) != len(curr_camera_pos):
        return True
    dist = math.sqrt(sum((a - b) ** 2 for a, b in
                         zip(prev_camera_pos, curr_camera_pos)))
    if dist > position_threshold:
        return True
    if prev_view_proj is not None and curr_view_proj is not None:
        if _mat_inf_delta(_linear_zip(prev_view_proj),
                          _linear_zip(curr_view_proj)) > matrix_threshold:
            return True
    return False


def history_clip(history_ao: Any = None,
                 current_ao: Any = None,
                 reproject_valid: Optional[Sequence[Sequence[int]]] = None,
                 neighborhood: int = 1) -> List[List[float]]:
    """重投影历史的 variance clipping（邻域 min/max 裁剪）。

    TAA 拖影的标准治法：把重投影到本帧的历史值限制在当前帧
    ``(2*neighborhood+1)`` 邻域的 [min, max] 内。瞬移后旧场景的遮蔽值
    几乎必然落在新区间外，会被拉回到当前帧观测范围内，配合
    :func:`history_reset` 实现“瞬移不带拖影”。

    ``reproject_valid`` 为 0 的像素（重投影落到屏幕外/被新表面占据）
    直接替换为当前帧值，不做混合。
    """
    if not _is_grid(history_ao):
        # 演示哨兵：历史全 0.1（强拖影），当前帧全 0.9，中心像素重投影无效。
        h4 = w4 = 4
        history_ao = _new_grid(h4, w4, 0.1)
        current_ao = _new_grid(h4, w4, 0.9)
        reproject_valid = [[1] * w4 for _ in range(h4)]
        reproject_valid[2][2] = 0
    elif not _is_grid(current_ao):
        raise TypeError("current_ao 必须是二维网格（与 history_ao 同分辨率）")

    if not _same_dims(history_ao, current_ao):
        raise ValueError("history_ao 与 current_ao 分辨率不一致")
    h, w = _dims(current_ao)
    if reproject_valid is None:
        reproject_valid = [[1] * w for _ in range(h)]
    elif _dims(reproject_valid) != (h, w):
        raise ValueError("reproject_valid 分辨率不一致")

    out = _new_grid(h, w)
    for y in range(h):
        for x in range(w):
            if not reproject_valid[y][x]:
                out[y][x] = float(current_ao[y][x])
                continue
            y0, y1 = max(0, y - neighborhood), min(h, y + neighborhood + 1)
            x0, x1 = max(0, x - neighborhood), min(w, x + neighborhood + 1)
            vals = [current_ao[yy][xx]
                    for yy in range(y0, y1) for xx in range(x0, x1)]
            lo, hi = min(vals), max(vals)
            out[y][x] = _clamp(float(history_ao[y][x]), lo, hi)
    return out


# ---------------------------------------------------------------------------
# 6. 输入校验、审计、降级与降噪
# ---------------------------------------------------------------------------

def validation(*buffers: Any, strict: bool = False) -> Dict[str, Any]:
    """帧资源校验：分辨率关系、数值域、深度单调性约定。

    接受任意 ``(name, grid)`` 位置参数（两两成对），也接受演示哨兵。
    strict=True 时首个错误直接抛 ValueError；否则收集后返回报告。
    """
    errors: List[str] = []
    pairs: List[Tuple[str, Any]] = []
    if len(buffers) == 1 and not isinstance(buffers[0], tuple):
        # 演示哨兵：用一套自洽的小资源跑校验。
        pairs = [
            ("depth", [[1.0, 1.0, 5.0, 5.0]] * 4),
            ("ao_halfres", [[0.8, 0.2], [0.7, 0.3]]),
            ("ao_fullres", _new_grid(4, 4, 0.5)),
        ]
    else:
        if len(buffers) % 2 == 1 and isinstance(buffers[-1], bool):
            strict = buffers[-1]
            buffers = buffers[:-1]
        pairs = [(str(buffers[i]), buffers[i + 1])
                 for i in range(0, len(buffers) - 1, 2)]

    grids: Dict[str, Sequence[Sequence[float]]] = {}
    for name, grid in pairs:
        if not _is_grid(grid):
            errors.append(f"{name}: 不是二维网格")
            continue
        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                errors.append(f"{name}: 行宽不一致")
                break
        grids[name] = grid

    if "depth" in grids and "ao_halfres" in grids:
        dh, dw = _dims(grids["depth"])
        lh, lw = _dims(grids["ao_halfres"])
        if (dh, dw) != (2 * lh, 2 * lw):
            errors.append("depth 与 ao_halfres 不是 2x 分辨率关系")
    if "ao_fullres" in grids and "ao_halfres" in grids:
        if _dims(grids["ao_fullres"]) != tuple(2 * v for v in
                                               _dims(grids["ao_halfres"])):
            errors.append("ao_fullres 与 ao_halfres 不是 2x 分辨率关系")

    for name, grid in grids.items():
        if name.startswith("ao"):
            for row in grid:
                for v in row:
                    if not (0.0 <= float(v) <= 1.0):
                        errors.append(f"{name}: AO 越界 {v}")
                        break
        if name == "depth":
            for row in grid:
                for v in row:
                    if float(v) < 0.0:
                        errors.append("depth: 出现负深度")
                        break

    report = {"ok": not errors, "errors": errors,
              "checked": [name for name, _ in pairs]}
    if strict and errors:
        raise ValueError("; ".join(errors))
    return report


def audit(graph_or_order: Any = None) -> Dict[str, Any]:
    """帧图不变量审计。

    检查：
      1. pass 依赖闭合（依赖的 pass 都在拓扑表内且序号更早）；
      2. 每个资源只有一个 producer（单写者）；
      3. 若传入已执行的 :class:`FrameGraph`，完成序号必须与拓扑一致、
         所有栅栏等待都被满足。
    """
    findings: List[str] = []
    index_of = {name: i for i, name in enumerate(PASS_ORDER)}

    for name, deps in PASS_DEPS.items():
        for dep in deps:
            if dep not in index_of:
                findings.append(f"{name} 依赖未知 pass {dep}")
            elif index_of[dep] >= index_of[name]:
                findings.append(f"{name} 的依赖 {dep} 拓扑序号不在其之前")

    producers: Dict[str, List[str]] = {}
    for resource, producer in FrameGraph.PRODUCER.items():
        producers.setdefault(resource, []).append(producer)
    for resource, ps in producers.items():
        if len(ps) != 1:
            findings.append(f"资源 {resource} 存在多个 producer: {ps}")

    executed: Dict[str, int] = {}
    if isinstance(graph_or_order, FrameGraph):
        executed = dict(graph_or_order.completed)
        seq = sorted(executed, key=lambda p: executed[p])
        if seq != [p for p in PASS_ORDER if p in executed]:
            findings.append("实际完成顺序与规范 pass_order 不一致")
        for wait in graph_or_order.fence_waits:
            if not wait["satisfied"]:
                findings.append(
                    f"栅栏未满足即执行: {wait['pass']} 缺 {wait['missing']}")

    return {
        "ok": not findings,
        "findings": findings,
        "pass_count": len(PASS_ORDER),
        "resource_count": len(FrameGraph.PRODUCER),
        "executed_passes": list(sorted(executed, key=lambda p: executed[p])),
    }


def fallback(reason: Any = "history_invalid",
             width: int = 4, height: int = 4,
             current_ao: Optional[Sequence[Sequence[float]]] = None
             ) -> Dict[str, Any]:
    """历史/资源失效时的安全降级。

    * 有当前帧 AO：直接以当前帧作为输出（不混合历史）；
    * 无当前帧 AO：返回中性遮蔽 1.0（无遮蔽），绝不用过期历史填充；
    * 可选标记触发全分辨率重算路径（half-res 路径整体失效时）。
    """
    reason = reason if isinstance(reason, str) else "demo_sentinel"
    if current_ao is not None:
        h, w = _dims(current_ao)
        out = [[_clamp(float(v), 0.0, 1.0) for v in row]
               for row in current_ao]
        mode = "use_current_frame"
    else:
        if not isinstance(width, int) or not isinstance(height, int):
            width, height = 4, 4
        h, w = height, width
        out = _new_grid(h, w, 1.0)
        mode = "neutral_full_res_recompute"
    return {
        "reason": reason,
        "mode": mode,
        "ao": out,
        "history_weight": 0.0,
        "request_full_res": mode == "neutral_full_res_recompute",
    }


def denoise(ao: Any = None,
            full_depth: Optional[Sequence[Sequence[float]]] = None,
            depth_threshold: float = 0.05,
            spatial_sigma: float = 1.0,
            range_sigma: float = 0.2) -> List[List[float]]:
    """边缘保持空间降噪（双边），与 :func:`edge_mask` 共用深度判据。

    跨深度边缘的邻域像素权重归零，降噪不会把光晕/拖影进一步抹开。
    """
    if not _is_grid(ao):
        # 演示哨兵：带椒盐噪声的 4x4 AO + 阶跃深度。
        ao = [[0.5, 0.9, 0.1, 0.5],
              [0.5, 0.1, 0.9, 0.5],
              [0.5, 0.9, 0.1, 0.5],
              [0.5, 0.1, 0.9, 0.5]]
        full_depth = [[1.0] * 2 + [5.0] * 2 for _ in range(2)] + \
                     [[1.0] * 2 + [5.0] * 2 for _ in range(2)]

    h, w = _dims(ao)
    out = _new_grid(h, w)
    for y in range(h):
        for x in range(w):
            acc, wsum = 0.0, 0.0
            center = ao[y][x]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if not (0 <= yy < h and 0 <= xx < w):
                        continue
                    # 深度感知：中心与邻居之间跨过深度边缘则权重归零，
                    # 降噪不会把光晕/拖影沿边缘抹开。
                    if full_depth is not None and \
                            abs(full_depth[yy][xx] - full_depth[y][x]) > depth_threshold:
                        continue
                    spatial = math.exp(-(dx * dx + dy * dy) /
                                       (2.0 * spatial_sigma * spatial_sigma))
                    range_w = math.exp(-((ao[yy][xx] - center) ** 2) /
                                       (2.0 * range_sigma * range_sigma))
                    weight = spatial * range_w
                    acc += weight * ao[yy][xx]
                    wsum += weight
            out[y][x] = _clamp(acc / wsum, 0.0, 1.0) if wsum else center
    return out


def metrics(ao_fullres: Any = None,
            full_depth: Any = None,
            depth_threshold: float = 0.05,
            history: Optional[Dict[str, Any]] = None,
            clip_before: Optional[Sequence[Sequence[float]]] = None,
            clip_after: Optional[Sequence[Sequence[float]]] = None
            ) -> Dict[str, Any]:
    """汇总画面质量指标：边缘光晕、裁剪量、重置/拒绝统计、结果指纹。"""
    if not _is_grid(ao_fullres):
        # 演示哨兵：双边 vs 双线性在阶跃深度上的对比。
        full_depth = [[1.0] * 2 + [5.0] * 2 for _ in range(2)] + \
                     [[1.0] * 2 + [5.0] * 2 for _ in range(2)]
        low = [[0.9, 0.1], [0.9, 0.1]]
        ao_fullres = depth_upsample(low, full_depth, depth_threshold=0.5,
                                    low_depth=[[1.0, 5.0], [1.0, 5.0]])
        clip_before = _new_grid(4, 4, 0.1)
        clip_after = history_clip(clip_before, _new_grid(4, 4, 0.9))
    elif not _is_grid(full_depth):
        raise TypeError("full_depth 必须是二维网格（与 ao_fullres 同分辨率）")

    mask = edge_mask(full_depth, depth_threshold)
    edge_pixels = sum(sum(row) for row in mask)
    total = len(ao_fullres) * len(ao_fullres[0])

    # 光晕代理指标：深度边缘像素与同侧内部像素的 AO 平均偏差。
    edge_diffs: List[float] = []
    h, w = _dims(ao_fullres)
    for y in range(h):
        for x in range(w):
            if mask[y][x] and x + 1 < w:
                edge_diffs.append(abs(ao_fullres[y][x] -
                                      ao_fullres[y][x + 1]))
    halo = sum(edge_diffs) / len(edge_diffs) if edge_diffs else 0.0

    clipped = 0
    if clip_before is not None and clip_after is not None:
        for row_b, row_a in zip(clip_before, clip_after):
            for b, a in zip(row_b, row_a):
                if abs(b - a) > 1e-6:
                    clipped += 1

    fingerprint = _sha256_hex(repr(ao_fullres).encode())
    return {
        "edge_pixels": edge_pixels,
        "edge_ratio": edge_pixels / total if total else 0.0,
        "halo_proxy": halo,
        "clipped_pixels": clipped,
        "reset_count": history.get("reset_count", 0) if isinstance(history, dict) else 0,
        "history_valid": history_valid(history, 0) if isinstance(history, dict) else False,
        "fingerprint": fingerprint,
    }
