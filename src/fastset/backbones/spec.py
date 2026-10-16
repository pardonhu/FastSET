"""纯Python参数预算：只按完整模型计数选宽度，不训练、不查看BER。"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Spec:
    family: str
    role: str
    depth: int
    width: int
    heads: int = 4
    token_hidden: int = 32

    def __post_init__(self):
        if self.family not in ("transformer", "mixer") or self.role not in ("e2e", "est", "dem"):
            raise ValueError("未知backbone或角色")
        if (
            self.depth < 1
            or self.width < 4
            or self.width % 4
            or self.heads != 4
            or self.token_hidden < 1
        ):
            raise ValueError("深度/宽度/heads配置非法")

    def dictionary(self):
        return asdict(self)

    @property
    def inputs(self):
        return 7 if self.role == "dem" else 5

    @property
    def outputs(self):
        return 2 if self.role == "est" else 4

    def parameters(self):
        d = self.width
        # 2x6=12RE/token；固定位置编码没有可训练参数。
        io = (12 * self.inputs + 1) * d + (d + 1) * 12 * self.outputs + 2 * d
        if self.family == "transformer":
            block = 8 * d * d + 11 * d  # 两LN，Q/K/V/out投影，FFN=2D，均含bias。
        else:
            block = 4 * d * d + 7 * d + 2 * 308 * self.token_hidden + self.token_hidden + 308
        return io + self.depth * block


# 最新用户优先级：总参数 > Est参数占比 > 深度/宽度比例。
REFERENCE_EST = 22434
REFERENCE_DEM = 636420
E2E_DEPTHS = tuple(range(4, 13))
EST_DEPTHS = tuple(range(1, 6))
EST_WIDTHS = tuple(range(16, 65, 4))
DEM_DEPTHS = tuple(range(4, 13))
WIDTHS = tuple(range(16, 257, 4))


def split_budget(
    family, budget, widths=WIDTHS, reference_est=REFERENCE_EST, reference_dem=REFERENCE_DEM
):
    """先满足同族99%-100%容量，再匹配Est占比，最后深宽比例。

    E2E选CNN预算以内最大可行容量；1%是预先固定的工程容差。
    在合格切分中先最小化Est占比误差，比例并列才比较深宽比例，
    然后用总参数及结构元组确定性破同分。不读取BER，不训练。
    """
    from fractions import Fraction

    if type(budget) is not int or budget <= 0:
        raise ValueError("budget必须正整数")
    if any(type(x) is not int or x <= 0 for x in (reference_est, reference_dem)):
        raise ValueError("参考参数量必须正整数")
    ratio = Fraction(reference_est, reference_est + reference_dem)
    e2es = [Spec(family, "e2e", d, w) for d in E2E_DEPTHS for w in widths]
    e2es = [s for s in e2es if s.parameters() <= budget]
    if not e2es:
        raise ValueError("预算不足以创建E2E")
    e2e = min(e2es, key=lambda s: (-s.parameters(), s.depth, s.width))
    candidates = []
    dems = [Spec(family, "dem", d, w) for d in DEM_DEPTHS for w in widths]
    for ed in EST_DEPTHS:
        for ew in EST_WIDTHS:
            est = Spec(family, "est", ed, ew, token_hidden=16)
            for dem in dems:
                total = est.parameters() + dem.parameters()
                if not 99 * e2e.parameters() <= 100 * total <= 100 * e2e.parameters():
                    continue
                ratio_error = abs(Fraction(est.parameters(), total) - ratio)
                depth_error = abs(Fraction(est.depth, dem.depth) - Fraction(1, 2))
                width_error = abs(Fraction(est.width, dem.width) - Fraction(1, 4))
                key = (
                    ratio_error,
                    depth_error + width_error,
                    depth_error,
                    width_error,
                    -total,
                    ed,
                    ew,
                    dem.depth,
                    dem.width,
                )
                candidates.append((key, est, dem))
    if not candidates:
        raise ValueError("同族99%-100%参数预算内无合法切分；不静默放宽容差")
    _, est, dem = min(candidates, key=lambda x: x[0])
    return dict(e2e=e2e, est=est, dem=dem)
