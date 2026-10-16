import hashlib, json, os
from pathlib import Path


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def filters(d, w, i):
    coe = 2 if d > 5 else 1
    return w * 2 ** ((i // coe) if i < d // 2 else ((d - i - 1) // coe))


def params(d, w, inputs, outputs):
    total = 9 * inputs * w + w
    cin = w
    for i in range(d):
        out = filters(d, w, i)
        total += 11 * cin + 2 * cin * out + out * out + 14 * out
        cin = out
    return total + 9 * cin * outputs + outputs


def architecture(d, w):
    est = params(d, w, 5, 2)
    budget = 661188
    remaining = budget - est
    dd = max(int(11 * remaining / budget), 1)
    lo, hi = 8, 512
    best, err = 64, float("inf")
    while lo <= hi:
        mid = (lo + hi) // 2
        n = params(dd, mid, 7, 4)
        e = abs(n - remaining) / remaining
        if n < remaining:
            if e < err:
                best, err = mid, e
            if e < 0.05:
                break
        if n >= remaining:
            hi = mid - 1
        else:
            lo = mid + 1
    dw = best // 4 * 4
    dem = params(dd, dw, 7, 4)
    assert est + dem <= budget
    return dict(
        id=f"d{d}_w{w}",
        depth=d,
        width=w,
        dem_depth=dd,
        dem_width=dw,
        est_parameters=est,
        dem_parameters=dem,
        parameters=est + dem,
    )


def order(n, epochs, seed=42):
    import numpy as np

    rng = np.random.RandomState(seed)
    paths = list(range(n))
    result = []
    for _ in range(epochs):
        rng.shuffle(paths)
        result.append(paths.copy())
    return result
