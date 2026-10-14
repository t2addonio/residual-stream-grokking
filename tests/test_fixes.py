"""Cheap CPU checks for the audit fixes (no training)."""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modarith_split import make_split  # noqa: E402
from dyck_outcomes import classify  # noqa: E402


def test_modarith_split_disjoint_and_complete():
    p = 13
    (xtr, ytr), (xte, yte) = make_split(p, seed=0, train_frac=0.3, device="cpu")
    tr = {tuple(r) for r in xtr[:, :2].tolist()}
    te = {tuple(r) for r in xte[:, :2].tolist()}
    assert not tr & te and len(tr | te) == p * p and len(tr) == int(p * p * 0.3)
    assert torch.equal(ytr, (xtr[:, 0] + xtr[:, 1]) % p)
    (xtr2, _), _ = make_split(p, seed=0, train_frac=0.3, device="cpu")
    assert torch.equal(xtr, xtr2)


def test_dyck_heldout_shuffled_and_balanced_subset():
    import train_dyck_30seed as T
    _, (x_ho, y_ho) = T.get_train_heldout(100)
    assert 0.35 < y_ho[:1024].float().mean().item() < 0.65  # was 0.977 before the shuffle
    xs, ys = T.balanced_subset(x_ho, y_ho, 512, 0)
    assert len(ys) == 1024 and ys.sum().item() == 512


def test_dyck_metric_ignores_pad_logit():
    import train_dyck_30seed as T
    torch.manual_seed(0)
    m = T.make_model("cpu")
    with torch.no_grad():
        m.b_U[2] += 1e4  # make PAD win the 3-way argmax everywhere
    x = torch.randint(0, 2, (64, 16))
    y = torch.ones(64, dtype=torch.long)
    st = T.prediction_stats(m, x, y)
    assert st["pad_argmax_rate"] == 1.0
    # old metric would report every example as class 0; new one uses the class logits only
    preds = m(x)[:, -1, :2].argmax(-1).float()
    assert abs(st["pred_pos_rate"] - preds.mean().item()) < 1e-6


def test_classify_outcomes():
    assert classify(0.01, 0.49, 0.99) == "label_flip"
    assert classify(0.50, 0.0, 0.99) == "single_class"
    assert classify(0.50, None, 0.99) == "chance_ambiguous"
    assert classify(0.99, 0.5, 0.99) == "intact"
    assert classify(0.70, 0.5, 0.99) == "partial"


def test_infer_dims_any_width():
    import fourier_overlap_test as F
    for d, h, dh, mlp in [(128, 4, 32, 512), (256, 4, 64, 1024)]:
        m = F.make_model(0, d_model=d, n_heads=h, d_head=dh, d_mlp=mlp)
        assert F.infer_dims(m.state_dict()) == dict(d_model=d, n_heads=h, d_head=dh, d_mlp=mlp, n_layers=2)


def test_analyze_dyck_exits_nonzero_without_seeds(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "analyze_dyck_directions.py"),
                        "--results_dir", str(tmp_path), "--out_dir", str(tmp_path / "out")],
                       cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode != 0 and "no seed_*.pt" in r.stderr

def test_subspace_overlap_bounded_and_calibrated():
    import train_dyck_30seed as T
    torch.manual_seed(0)
    d, k = 64, 8
    Q = torch.linalg.qr(torch.randn(d, d))[0]
    A = Q[:, :k].T
    B_same = Q[:, :k].T
    B_orth = Q[:, k:k + 3].T
    # Old bug pattern: duplicate rows, only row-normalized. Prior metric
    # returned 8.0 here; principal-angle overlap must stay in [0, 1].
    dup = torch.stack([A[0], A[0], A[1], A[1], A[2], A[3], A[4], A[5]])
    assert abs(T.subspace_overlap(A, B_same) - 1.0) < 1e-5
    assert T.subspace_overlap(A, B_orth) < 1e-5
    assert 0.0 <= T.subspace_overlap(dup, A) <= 1.0 + 1e-5
    # Rank-3 inside rank-8 -> 1.0
    assert abs(T.subspace_overlap(A, A[:3]) - 1.0) < 1e-5
