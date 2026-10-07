"""拼图机：把图片切成碎片并随机打乱，再对照原图，靠碎片内容与形状精确复原。
支持两种切割：jigsaw（默认，边缘凸头/凹槽咬合）与 square（方块直切，像素级无损）。

设计：
  scramble：原图 → R×C 网格骨架 → 每条内部缝生成互补的圆润凸头(tab)/凹槽(blank)
            → 每块为带透明蒙版的 RGBA，bounding box 随凸头超出格子 → 打乱/旋转成列
  solve   ：对照原图，碎片在整图上滑动，只在不透明像素内做归一化匹配(NCC)，
            结合4个旋转与全局一一分配 → alpha 合成复原

GUI：python puzzle_machine.py
命令行：
  python puzzle_machine.py scramble <原图> [块大小=96] [输出目录] [--rotate] [--square]
  python puzzle_machine.py solve <碎片目录> <原图> <复原图> [顺序表CSV]
  python puzzle_machine.py solve-noref <碎片目录> [输出目录]   # 无原图，仅凭咬合碎片
"""
import csv
import json
import math
import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

PACK_NAME = "pack.json"
ANSWER_NAME = "answer_key.json"
SUPER = 4  # 超采样倍数（抗锯齿）


# ==================== 凸头形状 ====================

def _tab_polygon(length):
    """生成“经典圆润对称”凸头轮廓点（沿 +方向），单位像素（普通尺度）。
    凸头根部位于缝上 (0,0)，向 +x 鼓出。轮廓为对称闭合多边形。
    返回的点位于 x>=0 侧，配合垂直/水平缝时再做坐标映射。
    """
    L = length
    anchor = L * 0.08        # 根部锚进缝内侧的深度（保证凹槽被完整挖出）
    neck = L * 0.125         # 颈部半宽
    head_r = L * 0.17        # 圆头半径
    head_x = L * 0.08        # 圆头中心相对缝的位置（鼓出总长 head_x+head_r ≈ 0.25L）
    pts = []
    pts.append((-anchor, -neck))
    pts.append((L * 0.02, -neck))
    pts.append((L * 0.04, -head_r * 0.95))
    pts.append((L * 0.06, -head_r))
    for k in range(13):
        a = -math.pi / 2 + math.pi * k / 12
        pts.append((head_x + head_r * math.cos(a), head_r * math.sin(a)))
    pts.append((L * 0.06, head_r))
    pts.append((L * 0.04, head_r * 0.95))
    pts.append((L * 0.02, neck))
    pts.append((-anchor, neck))
    return pts


def _map_tab(poly, orientation, origin):
    """把基准凸头多边形映射到某条缝上。
    orientation: 'v'(竖直缝,凸头沿+x) / 'vneg'(-x) / 'h'(水平缝,凸头沿+y) / 'hneg'(-y)
    origin: (ox,oy) 根部中心在网格画布中的坐标。
    """
    ox, oy = origin
    out = []
    for x, y in poly:
        if orientation == "v":
            p = (ox + x, oy + y)
        elif orientation == "vneg":
            p = (ox - x, oy + y)
        elif orientation == "h":
            p = (ox + y, oy + x)
        else:  # hneg
            p = (ox + y, oy - x)
        out.append(p)
    return out


# ==================== 打乱 ====================

def _mirror_pad(img, target_w, target_h):
    orig_w, orig_h = img.size
    canvas = Image.new("RGB", (target_w, target_h))
    canvas.paste(img, (0, 0))
    if target_w > orig_w:
        strip = img.crop((max(0, 2 * orig_w - target_w), 0, orig_w, orig_h))
        canvas.paste(strip.transpose(Image.FLIP_LEFT_RIGHT), (orig_w, 0))
    if target_h > orig_h:
        strip = img.crop((0, max(0, 2 * orig_h - target_h), orig_w, orig_h))
        canvas.paste(strip.transpose(Image.FLIP_TOP_BOTTOM), (0, orig_h))
    if target_w > orig_w and target_h > orig_h:
        corner = img.crop((max(0, 2 * orig_w - target_w), max(0, 2 * orig_h - target_h),
                           orig_w, orig_h))
        corner = corner.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
        canvas.paste(corner, (orig_w, orig_h))
    return canvas


def scramble(src_path, out_dir, block_size=96, keep_answer=True, seed=None, rotate=False,
             style="jigsaw"):
    """生成带凸头/凹槽的不规则碎片并随机打乱成列。"""
    import random
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"找不到原图：{src_path}")
    if block_size < 40:
        raise ValueError("真拼图块大小建议 >= 40 像素")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    base = Image.open(src_path).convert("RGB")
    orig_w, orig_h = base.size
    cols, rows = math.ceil(orig_w / block_size), math.ceil(orig_h / block_size)
    W, H = cols * block_size, rows * block_size
    if (W, H) != (orig_w, orig_h):
        base = _mirror_pad(base, W, H)
    ref = np.asarray(base, dtype="uint8")

    # 内部缝的归属：'vertical'[(r,c)] 表示 r 行、位于 cell c 与 c+1 之间的竖缝
    # owner=1 凸头归左块(+x在左块里)；owner=0 凸头归右块(凸头沿-x)
    vowner = {}
    for r in range(rows):
        for c in range(cols - 1):
            vowner[(r, c)] = 1 if rng.random() < 0.5 else 0
    howner = {}
    for r in range(rows - 1):
        for c in range(cols):
            howner[(r, c)] = 1 if rng.random() < 0.5 else 0

    poly = _tab_polygon(block_size)

    # 蒙版画布：咬合用4倍超采样抗锯齿；方块直切直接1倍，保证像素级无损
    s = SUPER if style == "jigsaw" else 1

    def new_mask():
        return Image.new("L", (W * s, H * s), 0)

    piece_records = []  # (row,col, mask普通尺度array, bbox(normal), crop_x,crop_y)
    for pr in range(rows):
        for pc in range(cols):
            m = new_mask()
            d = ImageDraw.Draw(m)
            x0, y0 = pc * block_size, pr * block_size
            x1, y1 = x0 + block_size, y0 + block_size
            # 基础格子
            d.rectangle((x0 * s, y0 * s, x1 * s - 1, y1 * s - 1), fill=255)

            def add_tab(orientation, origin, positive):
                pts = [(px * s, py * s) for px, py in _map_tab(poly, orientation, origin)]
                ImageDraw.Draw(m).polygon(pts, fill=255 if positive else 0)
            if style == "jigsaw":
                if pc > 0:
                    own = vowner[(pr, pc - 1)]
                    add_tab("vneg", (x0, (y0 + y1) // 2), positive=(own == 0))
                if pc < cols - 1:
                    own = vowner[(pr, pc)]
                    add_tab("v", (x1, (y0 + y1) // 2), positive=(own == 1))
                if pr > 0:
                    own = howner[(pr - 1, pc)]
                    add_tab("hneg", ((x0 + x1) // 2, y0), positive=(own == 0))
                if pr < rows - 1:
                    own = howner[(pr, pc)]
                    add_tab("h", ((x0 + x1) // 2, y1), positive=(own == 1))


            # 缩回普通尺度（咬合：区域滤波式抗锯齿；方块：原样）
            mask_small = m.resize((W, H), Image.LANCZOS) if s > 1 else m
            ma = np.asarray(mask_small)
            ys, xs = np.where(ma > 12)
            cx0, cy0, cx1, cy1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            mask_crop = ma[cy0:cy1, cx0:cx1]
            piece_records.append({"row": pr, "col": pc,
                                  "mask": mask_crop, "crop": (cx0, cy0, cx1, cy1)})

    count = rows * cols
    # 为每块合成 RGBA（RGB 取自原图相应位置，alpha 为蒙版）
    pieces_rgba = []
    for rec in piece_records:
        cx0, cy0, cx1, cy1 = rec["crop"]
        rgb = ref[cy0:cy1, cx0:cx1, :].astype("float32")
        a = (rec["mask"].astype("float32") / 255.0)[..., None]
        rgba = np.dstack([rgb * a, rec["mask"]]).astype("uint8")  # 透明处RGB置0
        pieces_rgba.append(rgba)

    correct = [(rec["row"] + 1, rec["col"] + 1) for rec in piece_records]
    order = list(range(count))
    rng.shuffle(order)
    rotations = {i: (rng.choice((0, 90, 180, 270)) if rotate else 0) for i in range(count)}

    saved_pieces = []
    for col_idx, piece_idx in enumerate(order, 1):
        deg = rotations[piece_idx]
        src_rgba = pieces_rgba[piece_idx]
        rgba_img = Image.fromarray(src_rgba, "RGBA")
        ph0, pw0 = src_rgba.shape[:2]
        cx0, cy0 = piece_records[piece_idx]["crop"][0], piece_records[piece_idx]["crop"][1]
        t = deg // 90
        if deg:
            rgba_img = rgba_img.rotate(-deg, expand=True)
        # 文件内 nominal 偏移（未旋转紧凑块）：全局格点 - bbox起点
        fcx = piece_records[piece_idx]["col"] * block_size - cx0
        fcy = piece_records[piece_idx]["row"] * block_size - cy0
        vmeta = _rotate_piece_variant(src_rgba, fcx, fcy, block_size, block_size, t)
        noff = [vmeta["no_x"], vmeta["no_y"]]
        fn = f"pic{col_idx:03d}.png"
        rgba_img.save(os.path.join(out_dir, fn))
        saved_pieces.append({"file": fn, "crop": [cx0, cy0],
                             "nominal_offset_in_file": noff})

    pack = {"block_size": block_size, "cols": cols, "rows": rows, "count": count,
            "orig_width": orig_w, "orig_height": orig_h, "rotated": bool(rotate),
            "padded_width": W, "padded_height": H,
            "cut_style": style,
            "style": ("classic rounded symmetric tabs; straight outer border"
                      if style == "jigsaw" else "plain square blocks"),
            "pieces": saved_pieces,
            "note": ("不规则咬合碎片" if style == "jigsaw" else "按方块直切") +
                    "，picNNN.png 为打乱后的列顺序",
            "created_at": datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(out_dir, PACK_NAME), "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)

    if keep_answer:
        answer = [{"column_piece": i + 1, "file": f"pic{i + 1:03d}.png",
                   "row": correct[piece_idx][0], "col": correct[piece_idx][1],
                   "rotation": rotations[piece_idx]}
                  for i, piece_idx in enumerate(order)]
        with open(os.path.join(out_dir, ANSWER_NAME), "w", encoding="utf-8") as f:
            json.dump(answer, f, ensure_ascii=False, indent=2)
    return pack


# ==================== 求解 ====================

def _gray(rgb):
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])


def _rotate_piece_variant(rgba, cx0, cy0, nw, nh, t):
    """把 w0×h0 的碎片（nominal 格子在其内部 (cx0,cy0) 起、nw×nh 大小）
    顺时针旋转 90*t 度，返回紧凑裁剪后的 dict（alpha/gray + nominal 偏移）。
    """
    h0, w0 = rgba.shape[:2]
    if t:
        rgba = np.rot90(rgba, k=(-t) % 4)
    alpha = rgba[..., 3]
    yy, xx = np.where(alpha > 12)
    bx0, by0, bx1, by1 = int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1

    if t:
        H0, W0 = h0, w0
        def rp(x, y):
            k = (-t) % 4
            if k == 3:
                return H0 - 1 - y, x
            if k == 2:
                return W0 - 1 - x, H0 - 1 - y
            return y, W0 - 1 - x
        xs = [rp(cx0, cy0)[0], rp(cx0 + nw - 1, cy0)[0],
              rp(cx0, cy0 + nh - 1)[0], rp(cx0 + nw - 1, cy0 + nh - 1)[0]]
        ys = [rp(cx0, cy0)[1], rp(cx0 + nw - 1, cy0)[1],
              rp(cx0, cy0 + nh - 1)[1], rp(cx0 + nw - 1, cy0 + nh - 1)[1]]
        nom_x0, nom_y0 = min(xs), min(ys)
    else:
        nom_x0, nom_y0 = cx0, cy0

    # nominal 在紧凑 bbox 内的偏移（可能为正，因凸头超出格子）
    no_x, no_y = nom_x0 - bx0, nom_y0 - by0
    ac = alpha[by0:by1, bx0:bx1]
    gc = _gray(rgba[by0:by1, bx0:bx1, :3])
    return {"alpha": ac, "gray": gc, "w": bx1 - bx0, "h": by1 - by0,
            "no_x": no_x, "no_y": no_y, "nw": nw, "nh": nh}


def solve_puzzle(piece_dir, original_path, out_image, report_csv=None, progress_cb=None):
    """对照原图，滑动匹配不规则碎片（含旋转），全局一一分配后 alpha 合成复原。"""
    from scipy.optimize import linear_sum_assignment

    pack_path = os.path.join(piece_dir, PACK_NAME)
    if not os.path.isfile(pack_path):
        raise FileNotFoundError(f"找不到 {PACK_NAME}，请先执行打乱")
    with open(pack_path, encoding="utf-8") as f:
        pack = json.load(f)
    if not os.path.isfile(original_path):
        raise FileNotFoundError(f"找不到原始图片：{original_path}")

    rows, cols = pack["rows"], pack["cols"]
    W, H = pack["padded_width"], pack["padded_height"]
    n = rows * cols
    files = sorted(f for f in os.listdir(piece_dir) if f.lower().endswith(".png"))
    if len(files) != n:
        raise ValueError(f"应有 {n} 个碎片，实际找到 {len(files)} 个 PNG")

    ref = Image.open(original_path).convert("RGB")
    if ref.size != (pack["orig_width"], pack["orig_height"]):
        raise ValueError("原图尺寸与打乱时记录的不一致，请确认是同一张原图")
    if ref.size != (W, H):
        ref = _mirror_pad(ref, W, H)
    ref_gray = _gray(np.asarray(ref, dtype="float32"))
    canvas64 = ref_gray.astype("float64")

    # 加载碎片（RGBA），构造4个旋转版本（统一画布旋转+裁剪，记录 nominal 偏移）
    piece_rots = []
    bs = block_size = pack["block_size"]
    saved_meta = pack["pieces"]
    for i, f in enumerate(files):
        rgba = np.asarray(Image.open(os.path.join(piece_dir, f)).convert("RGBA"), dtype="float32")
        fcx, fcy = saved_meta[i]["nominal_offset_in_file"]
        versions = [_rotate_piece_variant(rgba, fcx, fcy, bs, bs, t) for t in range(4)]
        piece_rots.append(versions)

    def _masked_sad_at(alpha, gray, top, left):
        """bbox 左上角置于画布 (top,left) 时，蒙版内 |canvas-gray| 之和。"""
        h, w = alpha.shape
        mask = alpha > 128
        region = canvas64[top:top + h, left:left + w]
        return float(np.abs(region - gray)[mask].sum())

    bs = pack["block_size"]
    INF = 1e18
    cost = np.full((n, n), INF, dtype="float64")
    chosen = {}
    for i in range(n):
        for t, v in enumerate(piece_rots[i]):
            h, w = v["h"], v["w"]
            for cr in range(rows):
                for cc in range(cols):
                    top = cr * bs - v["no_y"]
                    left = cc * bs - v["no_x"]
                    if top < 0 or left < 0 or top + h > H or left + w > W:
                        continue
                    sval = _masked_sad_at(v["alpha"], v["gray"], top, left)
                    cell = cr * cols + cc
                    if sval < cost[i, cell]:
                        cost[i, cell] = sval
                        chosen[(i, cell)] = (t, left, top)
        if progress_cb:
            progress_cb(i + 1, n)

    ri, ci = linear_sum_assignment(cost)
    assignment = {int(i): int(cell) for i, cell in zip(ri, ci)}

    # alpha 合成复原
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    placement = {}
    for i, cell in assignment.items():
        t, px, py = chosen[(i, cell)]
        v = piece_rots[i][t]
        # 取出裁剪后的 RGBA
        rgba = np.asarray(Image.open(os.path.join(piece_dir, files[i])).convert("RGBA"), dtype="uint8")
        rot = np.rot90(rgba, k=(-t) % 4)
        ys, xs = np.where(rot[..., 3] > 12)
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        tile = Image.fromarray(rot[by0:by1, bx0:bx1, :], "RGBA")
        canvas.alpha_composite(tile, (px - bx0, py - by0))
        cr, ccc = divmod(cell, cols)
        placement[i] = (cr + 1, ccc + 1, cell + 1, t)

    restored = canvas.crop((0, 0, pack["orig_width"], pack["orig_height"])).convert("RGB")
    out_dir = os.path.dirname(os.path.abspath(out_image))
    os.makedirs(out_dir, exist_ok=True)
    restored.save(out_image)

    report = []
    for k in range(n):
        r, c, gid, t = placement[k]
        applied_deg = ((4 - t) % 4) * 90
        v = piece_rots[k][t]
        npix = int((v["alpha"] > 128).sum())
        mean_err = float(cost[k, assignment[k]]) / max(npix, 1)
        report.append({"column_order": k + 1, "file": files[k], "row": r, "col": c,
                       "global_id": gid, "rotation": applied_deg,
                       "mean_pixel_error": round(mean_err, 3)})
    if report_csv:
        with open(report_csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["column_order", "file", "row", "col",
                                              "global_id", "rotation", "mean_pixel_error"])
            w.writeheader()
            w.writerows(report)

    accuracy = None
    ans_path = os.path.join(piece_dir, ANSWER_NAME)
    if os.path.isfile(ans_path):
        with open(ans_path, encoding="utf-8") as f:
            answer = json.load(f)
        truth = {a["column_piece"]: (a["row"], a["col"], a.get("rotation", 0)) for a in answer}

        def ok(rep):
            z = truth.get(rep["column_order"])
            return z is not None and z[:2] == (rep["row"], rep["col"]) and z[2] == rep["rotation"]

        accuracy = sum(1 for rep in report if ok(rep)) / n

    return {"report": report, "accuracy": accuracy, "rows": rows, "cols": cols,
            "mean_pixel_error": float(sum(rp["mean_pixel_error"] for rp in report) / n)}


# ==================== 无原图复原（仅凭咬合碎片） ====================

def _load_no_ref_solver():
    """加载 solver_no_ref 下的求解器；找不到目录则给出清晰提示。"""
    import importlib.util, sys
    here = os.path.dirname(os.path.abspath(__file__))
    sdir = os.path.join(here, "solver_no_ref")
    spath = os.path.join(sdir, "solver.py")
    if not os.path.isfile(spath):
        raise FileNotFoundError("找不到 solver_no_ref/solver.py，无原图复原模块缺失")
    if sdir not in sys.path:
        sys.path.insert(0, sdir)
    spec = importlib.util.spec_from_file_location("pm_noref_solver", spath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def solve_without_original(piece_dir, out_dir=None):
    """无原图复原：仅凭碎片目录（含 pack.json，须为咬合拼图）还原。
    在 out_dir 输出 solved.png、order.csv、result.json；out_dir 默认即碎片目录。
    返回 dict(rows,cols,confidence,mean_seam_cost,bad_seams,out_dir)。"""
    mod = _load_no_ref_solver()
    with open(os.path.join(piece_dir, PACK_NAME), encoding="utf-8") as f:
        pack = json.load(f)
    if pack.get("cut_style", "jigsaw") == "square":
        raise ValueError("无原图复原仅支持咬合拼图(jigsaw)，方块直切无形状线索、不可靠")
    tmp = os.path.join(out_dir, "_noref_tmp") if out_dir else os.path.join(piece_dir, "_noref_tmp")
    _, _, _, _, meta = mod.assemble(piece_dir, tmp)
    od = out_dir or piece_dir
    os.makedirs(od, exist_ok=True)
    os.replace(os.path.join(tmp, "restored.png"), os.path.join(od, "solved.png"))
    os.replace(os.path.join(tmp, "order.csv"), os.path.join(od, "order.csv"))
    os.replace(os.path.join(tmp, "result.json"), os.path.join(od, "result.json"))
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    return {"rows": meta["rows"], "cols": meta["cols"],
            "confidence": meta["confidence"], "mean_seam_cost": meta["mean_seam_cost"],
            "bad_seams": meta["bad_seams"], "out_dir": od}


# ==================== 图形界面 ====================

def launch_gui():
    import threading
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("真拼图机")
    root.geometry("620x440")

    src_var = tk.StringVar()
    pdir_var = tk.StringVar()
    orig_var = tk.StringVar()
    bs_var = tk.IntVar(value=96)
    rotate_var = tk.BooleanVar(value=False)
    style_var = tk.StringVar(value="jigsaw")
    out_var = tk.StringVar()
    csv_var = tk.StringVar()
    status_var = tk.StringVar(value="先生成不规则碎片并打乱，再对照原图求解")
    bar = ttk.Progressbar(root, maximum=100)

    def choose_src():
        p = filedialog.askopenfilename(title="选择原始图片",
                                       filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")])
        if p:
            src_var.set(p)
            orig_var.set(p)
            base = os.path.join(os.path.dirname(p),
                                os.path.splitext(os.path.basename(p))[0] + "_jigsaw")
            pdir_var.set(base)
            out_var.set(os.path.join(base, "solved.png"))
            csv_var.set(os.path.join(base, "order.csv"))

    def choose_pdir():
        p = filedialog.askdirectory(title="选择碎片文件夹")
        if p:
            pdir_var.set(p)
            out_var.set(os.path.join(p, "solved.png"))
            csv_var.set(os.path.join(p, "order.csv"))

    def choose_orig():
        p = filedialog.askopenfilename(title="选择用于对照的原始图片",
                                       filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp *.webp"), ("所有文件", "*.*")])
        if p:
            orig_var.set(p)

    def do_scramble():
        try:
            if not src_var.get():
                raise ValueError("请先选择原始图片")
            pack = scramble(src_var.get(), pdir_var.get(), int(bs_var.get()),
                            rotate=rotate_var.get(), style=style_var.get())
            extra = "，含随机旋转" if rotate_var.get() else ""
            kind = "咬合碎片" if style_var.get() == "jigsaw" else "方块碎片"
            status_var.set(f"已打乱：{pack['rows']}×{pack['cols']}，{pack['count']}块{kind}{extra}")
            bar.configure(value=0)
            messagebox.showinfo("完成", f"已生成 {pack['count']} 个{kind}：\n{pdir_var.get()}\n\n"
                                        "answer_key.json 是答案，发给别人前删除")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def do_solve():
        try:
            if not pdir_var.get():
                raise ValueError("请选择碎片文件夹")
            if not orig_var.get():
                raise ValueError("请选择用于对照的原始图片")

            def progress(done, total):
                root.after(0, lambda: (bar.configure(value=done * 100 // total),
                                       status_var.set(f"对照原图滑动匹配中 {done}/{total}")))

            def work():
                try:
                    res = solve_puzzle(pdir_var.get(), orig_var.get(),
                                       out_var.get(), csv_var.get() or None, progress)
                    acc_txt = f"\n位置+方向正确率 {res['accuracy']*100:.0f}%" if res["accuracy"] is not None else ""
                    root.after(0, lambda: (bar.configure(value=100),
                                           status_var.set(f"求解完成{acc_txt}"),
                                           messagebox.showinfo("完成",
                                                f"复原图：{out_var.get()}\n顺序表：{csv_var.get()}{acc_txt}")))
                except Exception as e:
                    root.after(0, lambda e=e: messagebox.showerror("错误", str(e)))

            threading.Thread(target=work, daemon=True).start()
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def do_solve_noref():
        try:
            if not pdir_var.get():
                raise ValueError("请选择碎片文件夹")
            status_var.set("无原图复原中：几何定位+接缝匹配…")

            def work():
                try:
                    res = solve_without_original(pdir_var.get())
                    root.after(0, lambda: (bar.configure(value=100),
                                           status_var.set(f"无原图复原完成，置信度 {res['confidence']:.2f}"),
                                           messagebox.showinfo("完成",
                                                f"复原图：{pdir_var.get()}/solved.png\n"
                                                f"顺序表：{pdir_var.get()}/order.csv\n"
                                                f"置信度 {res['confidence']:.2f}，坏接缝 {res['bad_seams']} 处")))
                except Exception as e:
                    root.after(0, lambda e=e: messagebox.showerror("错误", str(e)))

            threading.Thread(target=work, daemon=True).start()
        except Exception as e:
            messagebox.showerror("错误", str(e))

    pad = {"padx": 8, "pady": 6}
    ttk.Label(root, text="① 原始图片：").grid(row=0, column=0, sticky="w", **pad)
    ttk.Entry(root, textvariable=src_var, width=54).grid(row=0, column=1, padx=4)
    ttk.Button(root, text="浏览…", command=choose_src).grid(row=0, column=2, padx=4)

    ttk.Label(root, text="块大小(像素)：").grid(row=1, column=0, sticky="w", **pad)
    ttk.Spinbox(root, from_=40, to=2048, textvariable=bs_var, width=8).grid(row=1, column=1, sticky="w", padx=4)
    ttk.Checkbutton(root, text="每块随机旋转 0/90/180/270", variable=rotate_var).grid(
        row=1, column=2, sticky="w", padx=4)

    style_box = ttk.Frame(root)
    style_box.grid(row=2, column=0, columnspan=3, sticky="w", **pad)
    ttk.Label(style_box, text="切割方式：").pack(side="left")
    ttk.Radiobutton(style_box, text="咬合拼图(凸凹)", variable=style_var,
                    value="jigsaw").pack(side="left", padx=8)
    ttk.Radiobutton(style_box, text="方块直切", variable=style_var,
                    value="square").pack(side="left", padx=8)

    ttk.Label(root, text="碎片文件夹：").grid(row=3, column=0, sticky="w", **pad)
    ttk.Entry(root, textvariable=pdir_var, width=54).grid(row=3, column=1, padx=4)
    ttk.Button(root, text="浏览…", command=choose_pdir).grid(row=3, column=2, padx=4)

    ttk.Label(root, text="② 求解对照原图：").grid(row=4, column=0, sticky="w", **pad)
    ttk.Entry(root, textvariable=orig_var, width=54).grid(row=4, column=1, padx=4)
    ttk.Button(root, text="浏览…", command=choose_orig).grid(row=4, column=2, padx=4)

    btns = ttk.Frame(root)
    btns.grid(row=5, column=0, columnspan=3, pady=12)
    ttk.Button(btns, text="打乱成列", command=do_scramble).pack(side="left", padx=16)
    ttk.Button(btns, text="对照原图求解", command=do_solve).pack(side="left", padx=16)
    ttk.Button(btns, text="无原图复原", command=do_solve_noref).pack(side="left", padx=16)

    bar.grid(row=6, column=0, columnspan=3, sticky="ew", padx=14)
    ttk.Label(root, textvariable=status_var, foreground="#555").grid(row=7, column=0, columnspan=3, pady=6)
    ttk.Label(root, text="咬合碎片边缘凸凹互补、外圈平直；方块直切为规整矩形；求解只在不透明像素内匹配，自动转正",
              foreground="#888").grid(row=8, column=0, columnspan=3)

    root.mainloop()


def main():
    import sys
    if len(sys.argv) == 1:
        launch_gui()
    elif sys.argv[1] == "scramble" and len(sys.argv) >= 3:
        args = sys.argv[2:]
        rot = "--rotate" in args
        square = "--square" in args
        args = [a for a in args if a not in ("--rotate", "--square")]
        src = args[0]
        bs = int(args[1]) if len(args) >= 2 else 96
        out = args[2] if len(args) >= 3 else os.path.splitext(src)[0] + (
            "_square" if square else "_jigsaw")
        p = scramble(src, out, bs, rotate=rot, style="square" if square else "jigsaw")
        tag = "（含随机旋转）" if rot else ""
        kind = "方块碎片" if square else "咬合碎片"
        print(f"完成：{p['rows']}×{p['cols']}，{p['count']} 块{kind}已打乱{tag} → {out}")
    elif sys.argv[1] == "solve" and len(sys.argv) >= 5:
        csv_path = sys.argv[5] if len(sys.argv) >= 6 else os.path.join(sys.argv[2], "order.csv")
        res = solve_puzzle(sys.argv[2], sys.argv[3], sys.argv[4], csv_path,
                           lambda d, t: print(f"\r{d}/{t}", end="", flush=True))
        acc = f"，正确率 {res['accuracy']*100:.0f}%" if res["accuracy"] is not None else ""
        print(f"\n完成 → {sys.argv[4]}，顺序表 → {csv_path}{acc}，平均每像素误差 {res['mean_pixel_error']:.3f}")
    elif sys.argv[1] == "solve-noref" and len(sys.argv) >= 3:
        out = sys.argv[3] if len(sys.argv) >= 4 else None
        res = solve_without_original(sys.argv[2], out)
        od = res["out_dir"]
        print(f"\n完成 → {od}/solved.png，顺序表 → {od}/order.csv，"
              f"置信度 {res['confidence']:.3f}，坏接缝 {res['bad_seams']} 处")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
