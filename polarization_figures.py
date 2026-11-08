import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

# ---------------- параметры ----------------
b = 0.35                    # амплитуда утечки поляризатора (преувеличена)
a = np.sqrt(1 - b**2)
theta_b = np.arcsin(b)      # угол отрезка после λ/4

DQ_DEG = 6.0                # ошибка λ/4 для рисунка 2, градусы
DH_DEG = 3.0                # ошибка λ/2 для рисунка 2, градусы
                            # (2*DH = DQ: поворот скомпенсирован,
                            #  остаётся только эллиптичность)

DQ2_DEG = 5.0               # ошибка λ/4 для схемы 2, градусы

COLOR_POL = "#534AB7"       # поляризация
COLOR_AN = "#D85A30"        # ось анализатора


# ---------------- матрицы Джонса ----------------
def R(phi):
    """Поворот на угол phi."""
    return np.array([[np.cos(phi), -np.sin(phi)],
                     [np.sin(phi),  np.cos(phi)]])


def Q(psi):
    """Четвертьволновая пластинка, быстрая ось под углом psi к p."""
    return R(psi) @ np.diag([1, 1j]) @ R(-psi)


def H(psi):
    """Полуволновая пластинка, быстрая ось под углом psi к p."""
    return R(psi) @ np.diag([1, -1]) @ R(-psi)


def trace(E, n=400):
    """Кривая, которую рисует конец вектора E(t) = Re[E e^{-i t}] за период."""
    t = np.linspace(0, 2 * np.pi, n)
    x = E[0].real * np.cos(t) + E[0].imag * np.sin(t)
    y = E[1].real * np.cos(t) + E[1].imag * np.sin(t)
    return x, y


# ---------------- рисование ----------------
def draw_panel(ax, E, caption, analyzer=None):
    """analyzer — угол оси анализатора к p в градусах (None — без анализатора)."""
    lim = 1.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")
    # оси p и s
    ax.plot([-lim, lim], [0, 0], color="0.6", lw=0.6)
    if analyzer is None:
        ax.plot([0, 0], [-lim, lim], color="0.6", lw=0.6)
        ax.text(0.06, lim * 0.92, "$s$", fontsize=11)
    else:
        al = np.radians(analyzer)
        ax.plot([-lim * np.cos(al), lim * np.cos(al)],
                [-lim * np.sin(al), lim * np.sin(al)],
                color=COLOR_AN, lw=1.4, ls="--")
        ax.text(lim * np.cos(al) + 0.06, lim * 0.92, "анализатор",
                color=COLOR_AN, fontsize=10)
    ax.text(lim * 0.95, 0.06, "$p$", fontsize=11)
    # поляризация
    x, y = trace(E)
    ax.plot(x, y, color=COLOR_POL, lw=2)
    ax.text(0, -lim - 0.18, caption, ha="center", va="top", fontsize=11)


def arrow_between(fig, ax_left, ax_right, label):
    p0, p1 = ax_left.get_position(), ax_right.get_position()
    y = (p0.y0 + p0.y1) / 2
    fig.patches.append(FancyArrowPatch(
        (p0.x1 + 0.005, y), (p1.x0 - 0.005, y), transform=fig.transFigure,
        arrowstyle="->", mutation_scale=16, color="0.3", lw=1.2))
    fig.text((p0.x1 + p1.x0) / 2, y + 0.04, label,
             ha="center", va="bottom", fontsize=12)


def make_figure(dq_deg, dh_deg, fname):
    dq, dh = np.radians(dq_deg), np.radians(dh_deg)
    E0 = np.array([a, -1j * b])             # после реального поляризатора
    E1 = Q(dq) @ E0                          # после λ/4
    E2 = H(theta_b / 2 + dh) @ E1            # после λ/2
    I = abs(E2[1]) ** 2                      # анализатор пропускает s-компоненту

    ideal = (dq_deg == 0 and dh_deg == 0)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2),
                             gridspec_kw={"wspace": 0.55})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.22)

    draw_panel(axes[0], E0, "после поляризатора\nэллипс $(a,\\,{-}ib)$")
    draw_panel(axes[1], E1,
               "после $\\lambda/4$\nотрезок под углом $\\theta_b$" if ideal
               else "после $\\lambda/4$\nпочти отрезок")
    draw_panel(axes[2], E2,
               "после $\\lambda/2$\nточно вдоль $p$: $I=0$" if ideal
               else "после $\\lambda/2$\nпрошло $I=" + f"{I:.4f}".replace(".", "{,}") + "$",
               analyzer=90)

    if ideal:   # дуга угла θ_b на средней панели
        axes[1].add_patch(Arc((0, 0), 1.0, 1.0, theta1=0,
                              theta2=np.degrees(theta_b), color=COLOR_POL, lw=1))
        axes[1].text(0.55, 0.12, "$\\theta_b$", color=COLOR_POL, fontsize=11)

    lq = "$\\lambda/4$" if ideal else f"$\\lambda/4$\n$\\delta\\psi_Q$ = {dq_deg:g}°"
    lh = "$\\lambda/2$" if ideal else f"$\\lambda/2$\n$\\delta\\psi_H$ = {dh_deg:g}°"
    arrow_between(fig, axes[0], axes[1], lq)
    arrow_between(fig, axes[1], axes[2], lh)

    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=200)
    plt.close(fig)
    print(f"{fname}: s-компонента на выходе = {E2[1]:.4f}, I = {I:.5f}")


def make_scheme2(rows, fname):
    """Схема 2: накачка вдоль p проходит λ/4 и λ/2 к образцу и обратно.

    rows — список строк (psi_Q, psi_H) или (psi_Q, psi_H, alpha) в градусах;
    alpha — угол оси анализатора к p (по умолчанию 90°, скрещён).
    Утечка поляризатора здесь не рисуется (вход — чистая p).
    """
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 3.6 * len(rows)),
                             gridspec_kw={"wspace": 0.55, "hspace": 0.55},
                             squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.12)
    for k, row in enumerate(rows):
        q_deg, h_deg = row[0], row[1]
        al_deg = row[2] if len(row) > 2 else 90
        q, h, al = np.radians(q_deg), np.radians(h_deg), np.radians(al_deg)
        E0 = np.array([1, 0], dtype=complex)
        E1 = Q(q) @ E0          # туда, после λ/4
        E2 = H(h) @ E1          # на образце
        E3 = H(h) @ E2          # обратно, после λ/2
        E4 = Q(q) @ E3          # обратно, после λ/4 -> к анализатору
        I = abs(np.cos(al) * E4[0] + np.sin(al) * E4[1]) ** 2   # проекция на ось анализатора
        ax = axes[k]
        draw_panel(ax[0], E1, "туда: после $\\lambda/4$")
        draw_panel(ax[1], E2, "на образце")
        draw_panel(ax[2], E3, "обратно: после $\\lambda/2$")
        draw_panel(ax[3], E4, "обратно: после $\\lambda/4$\nпрошло $I="
                   + f"{I:.3f}".replace(".", "{,}") + "$", analyzer=al_deg)
        label = f"$\\psi_Q$ = {q_deg:g}°\n$\\psi_H$ = {h_deg:g}°"
        if al_deg != 90:
            label += f"\n$\\alpha$ = {al_deg:g}°"
        fig.text(0.01, (ax[0].get_position().y0 + ax[0].get_position().y1) / 2,
                 label.replace(".", ","), va="center", fontsize=11)
        arrow_between(fig, ax[0], ax[1], "$\\lambda/2$")
        arrow_between(fig, ax[1], ax[2], "$\\lambda/2$")
        arrow_between(fig, ax[2], ax[3], "$\\lambda/4$")
        print(f"{fname}, psi_Q={q_deg}, psi_H={h_deg}, alpha={al_deg}: I = {I:.4f}")
    fig.savefig(fname + ".pdf")
    fig.savefig(fname + ".png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    make_figure(0, 0, "scheme1_ideal")
    make_figure(DQ_DEG, DH_DEG, "scheme1_errors")
    # схема 2: при psi_Q = 0 накачка возвращается вдоль p при любом psi_H
    make_scheme2([(0, 22.5), (0, 40)], "scheme2_ideal")
    # схема 2 с ошибкой λ/4 (преувеличена): вернувшаяся накачка повёрнута на 2*psi_Q;
    # во второй строке анализатор довёрнут на тот же угол, и утечка снова 0
    make_scheme2([(DQ2_DEG, 22.5), (DQ2_DEG, 22.5, 90 + 2 * DQ2_DEG)], "scheme2_errors")
