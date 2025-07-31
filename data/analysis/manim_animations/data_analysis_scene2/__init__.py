from manim import *
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress, kendalltau

from pathlib import Path

def import_trials_data(dataframe):
    df = dataframe.copy()
    study = 'Study_UID'
    condition = 'Condition_UID_In_Participant'
    participant = 'Participant_UID_In_Study'
    result = 'Trial_Participant_Response_Outcome'
    hits = 'Hit_Proportion'
    session = 'Session_UID_In_Participant'
    trials = 'Trial_UID_In_Session'

    df = df[df[study] == 2]
    df = df[df[condition] == '7']

    def calculate(dataframe, window=5):
        """
        moving mean no overlapping
        """
        df = dataframe.copy().sort_values([session, trials])
        df.loc[:, 'uid'] = range(1, len(df) + 1)
        df['tmp'] = (df['uid'] - 1)  // window
        df[hits] = df[result].groupby(df['tmp']).transform('mean')
        df = df.drop_duplicates(subset=['tmp'], keep='last')
        df.sort_values('uid', inplace=True)
        y = df[hits].tolist()
        x = df['uid'].tolist()
        return x, y

    d = {}
    for p, group in df.groupby(participant):
        x, y = calculate(group)

        window_size = 4
        moving_average = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
        # Calculate how many NaNs to pad (for odd window_size)
        n_pad = window_size - 1

        # Pad with NaNs at the beginning
        y = np.concatenate([
            np.full(n_pad, np.nan),
            moving_average])


        d[p] = {
            'i': range(1, len(x) + 1),
            'x': x,
            'y': y
        }

    return d

def import_scatter_plot_data(df):
    hits_df = df.pivot_table(
        index='participant',
        columns='group',
        values='result',
        aggfunc='sum'
    )

    count_df = df.pivot_table(
        index='participant',
        columns='group',
        values='index',
        aggfunc='count'
    )

    mean_df = df.pivot_table(
        index='participant',
        columns='group',
        values='index',
        aggfunc='mean'
    )

    hits_df = hits_df.rename(columns={''
        'left': 'left_hits',
        'right': 'right_hits',
        'teaching': 'teaching_hits'})
    hits_df['total_hits'] = (hits_df['left_hits'] + hits_df['right_hits'] + hits_df['teaching_hits'])

    count_df = count_df.rename(columns={
        'left': 'left_count',
        'right': 'right_count',
        'teaching': 'teaching_count'})
    count_df['total_count'] = (count_df['left_count'] + count_df['right_count'] + count_df['teaching_count'])

    delta_df = pd.concat([mean_df, count_df, hits_df], axis=1)
    delta_df['left_right_difference'] = delta_df['left'] - delta_df['right']
    delta_df['hit_proportion'] = delta_df['total_hits'] / delta_df['total_count']

    participants = delta_df.index.tolist()
    y = delta_df['hit_proportion'].to_list()
    x = delta_df['left_right_difference'].to_list()

    return participants, x, y


def import_eye_data(df_sorted):
    """
    ASSUMING DATA WAS SORTED BY NOVELTY LOOKING INDEX
    P10 0.13669587845617254
    P23 0.13214154597945166
    P17 0.12408289389417748
    P04 0.12136317827332382
    P14 0.10043986445632636
    P12 0.06683568878709656
    P03 0.06606801177809418
    P08 0.056899477393456854
    P15 0.0448244646953313
    P09 0.04226853570935846
    P07 0.02949607479730676
    P20 0.028896317257167348
    P26 0.023171756412890043
    P28 0.02086222832134066
    P25 0.01716699495066465
    P01 0.016033105985664375
    P16 0.00500025329854259
    P19 -0.00036771688901199306
    P05 -0.0051883895655314705
    P22 -0.011623274477576606
    P27 -0.01952880355604658
    P13 -0.04749919572972644
    """
    data = []
    for participant, cd_data in df_sorted.groupby('participant', sort=False):
        grouped_data = {}
        for group, group_data in cd_data.groupby('group', sort=False):
            if group_data.empty:
                raise ValueError(f"No data found for {participant}, {group}")

            group_data = group_data.sort_values(['session', 'trial_in_session'])

            trials = np.array([i+1 for i in range(len(group_data))])
            index = group_data['index'].tolist()

            # apply a moving average to smooth the data
            window_size = 4
            moving_average = np.convolve(index, np.ones(window_size)/window_size, mode='valid')
            # Calculate how many NaNs to pad (for odd window_size)
            n_pad = window_size - 1

            # Pad with NaNs at the beginning
            index = np.concatenate([
                np.full(n_pad, np.nan),
                moving_average])

            grouped_data[group] = {
                'x' : trials,
                'y' : index,
                'y_mean' : group_data['index'].mean()
            }
        data.append((participant, grouped_data))
    return data


def solid_dashed_plot(x_solid, x_dashed, y_solid, y_dashed, y_offset=0.0):

    mask_solid  = ~np.isnan(y_solid)
    mask_dashed = ~np.isnan(y_dashed)

    x_solid_clean  = np.array(x_solid)[mask_solid]
    y_solid_clean  = np.array(y_solid)[mask_solid]

    x_dashed_clean = np.array(x_dashed)[mask_dashed]
    y_dashed_clean = np.array(y_dashed)[mask_dashed]

    x_range = [min(x_solid), max(x_solid)+1, len(x_solid) // 7]
    y_range = [0.0, 1.01, 0.25]

    axes = Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=12,
        y_length=6,
        axis_config={"tip_width": 0.2, "tip_height": 0.2},
        tips=False
    ).shift(y_offset * DOWN)

    # axis labels
    x_label = Text("Tentativas", font_size=28)\
        .next_to(axes.x_axis.get_end(), RIGHT)

    y_label = Text("Olhar\nsílaba esquerda",
                   font_size=28,
                   line_spacing=0.8,
                   should_center=False)\
        .next_to(axes.y_axis.get_end(), UP)

    # tick labels
    x_ticks = VGroup(*[
        Text(str(i), font_size=20).next_to(axes.c2p(i, 0), DOWN, buff=0.2)
        for i in range(x_range[0], x_range[1], x_range[2])
    ])
    y_ticks = VGroup(*[
        Text(f"{i:.1f}", font_size=20).next_to(axes.c2p(0, i), LEFT, buff=0.2)
        for i in np.arange(y_range[0], y_range[1], y_range[2])
    ])

    # ------------------------------------------------------------------
    # solid line
    # ------------------------------------------------------------------
    solid_line = axes.plot(
        lambda t: np.interp(t, x_solid_clean, y_solid_clean),
        color=WHITE,
        stroke_width=4
    )

    # ------------------------------------------------------------------
    # dashed line (use DashedLine instead of DashedVMobject)
    # ------------------------------------------------------------------
    dashed_line = axes.plot(
        lambda t: np.interp(t, x_dashed_clean, y_dashed_clean),
        color=WHITE,
        stroke_width=3
    )
    dashed_line = DashedVMobject(
        dashed_line,
        num_dashes=len(x_dashed_clean),
        dashed_ratio=0.5)

    # ------------------------------------------------------------------
    # legend
    # ------------------------------------------------------------------
    legend = VGroup(
        Text("Palavra com sílaba nova", font_size=25),
        VGroup(
            Line(ORIGIN, RIGHT*0.5, color=WHITE, stroke_width=4),
            Text("à esquerda", font_size=25)
        ).arrange(RIGHT, buff=0.2),
        VGroup(
            DashedLine(ORIGIN, RIGHT*0.5, color=WHITE, stroke_width=2,
                        dash_length=1, dashed_ratio=0.5),
            Text("à direita", font_size=25)
        ).arrange(RIGHT, buff=0.2)
    ).arrange(DOWN, aligned_edge=LEFT)\
        .to_edge(UP)\
        .shift(y_offset * DOWN)

    group = VGroup(axes, x_label, y_label, x_ticks, y_ticks,
                    solid_line, dashed_line, legend)
    group.shift(y_offset * DOWN)
    group.diff_sum = float(np.average(y_solid) - np.average(y_dashed))
    return group

def single_solid_plot(x_solid, y_solid, y_offset=0.0):

    mask_solid  = ~np.isnan(y_solid)

    x_solid_clean  = np.array(x_solid)[mask_solid]
    y_solid_clean  = np.array(y_solid)[mask_solid]

    x_range = [min(x_solid), max(x_solid)+1, len(x_solid) // 7]
    y_range = [0.0, 1.01, 0.25]

    axes = Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=12,
        y_length=6,
        axis_config={"tip_width": 0.2, "tip_height": 0.2},
        tips=False
    ).shift(y_offset * DOWN)

    # axis labels
    x_label = Text("Tentativas", font_size=28).next_to(axes.x_axis.get_end(), RIGHT)
    y_label = Text("Acertos\n(proporção)",
                   font_size=28,
                   line_spacing=0.8,
                   should_center=False).next_to(
                       axes.y_axis.get_end(), UP)

    # tick labels
    x_ticks = VGroup(*[
        Text(str(i), font_size=20).next_to(axes.c2p(i, 0), DOWN, buff=0.2)
        for i in range(x_range[0], x_range[1], x_range[2])
    ])
    y_ticks = VGroup(*[
        Text(f"{i:.1f}", font_size=20).next_to(axes.c2p(0, i), LEFT, buff=0.2)
        for i in np.arange(y_range[0], y_range[1], y_range[2])
    ])

    # ------------------------------------------------------------------
    # solid line
    # ------------------------------------------------------------------
    solid_line = axes.plot(
        lambda t: np.interp(t, x_solid_clean, y_solid_clean),
        color=WHITE,
        stroke_width=4
    )

    group = VGroup(axes, x_label, y_label, x_ticks, y_ticks,
                    solid_line)
    group.shift(y_offset * DOWN)
    return group


def correlation_plot(data, y_offset=0.0):
    _, x, y = data
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # -------------------------------------------------
    # 1. Build axes
    # -------------------------------------------------
    x_range = [min(x), max(x), 0.05]
    y_range = [0.0, 1.01, 0.25]
    axes = Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=12,
        y_length=6,
        axis_config={"tip_width": 0.2, "tip_height": 0.2},
        tips=False
    ).shift(y_offset * DOWN)

    # -------------------------------------------------
    # 2. Labels
    # -------------------------------------------------
    x_label = Text("Índice de acompanhamento da sílaba nova",
                font_size=28) \
                .next_to(axes.x_axis.get_end(), RIGHT)

    y_label = Text("Índice de desempenho",
                font_size=28,
                line_spacing=0.8,
                should_center=False) \
                .next_to(axes.y_axis.get_end(), UP)

    # -------------------------------------------------
    # 3. Nice ticks (every 0.2 on both axes)
    # -------------------------------------------------

    x_ticks = VGroup(*[
        Text(f"{i:.2f}", font_size=20).next_to(axes.c2p(i, 0), DOWN, buff=0.2)
        for i in np.arange(x_range[0], x_range[1], x_range[2])
    ])
    y_ticks = VGroup(*[
        Text(f"{i:.1f}", font_size=20).next_to(axes.c2p(0, i), LEFT, buff=0.2)
        for i in np.arange(y_range[0], y_range[1], y_range[2])
    ])

    # -------------------------------------------------
    # 4. Scatter points
    # -------------------------------------------------
    points = VGroup(*[
        Dot(axes.c2p(xi, yi), color=BLUE, radius=0.1)
        for xi, yi in zip(x, y)
    ])

    # -------------------------------------------------
    # 5. Regression line
    # -------------------------------------------------
    A = np.vstack([x, np.ones_like(x)]).T

    r, r_p_value = pearsonr(x, y)
    tau, tau_p_value = kendalltau(x, y)

    slope, intercept, r_value, p_value, std_err = linregress(x, y, alternative='greater')

    x_line = np.linspace(x.min(), x.max(), 2)
    y_line = slope * x_line + intercept

    regression_line = Line(
        start=axes.c2p(x_line[0], y_line[0]),
        end=axes.c2p(x_line[1], y_line[1]),
        color=RED
    )

    # -------------------------------------------------
    # 6. pearson correlation, kendall tau
    # -------------------------------------------------
    stats = VGroup(
                MathTex(rf"r = {r:.3f}, p < 0.001"), # r_p_value < 0.001
                MathTex(rf"\tau = {tau:.3f}, p = {tau_p_value:.3f}"))\
            .arrange(DOWN, buff=0.2) \
            .to_corner(UR, buff=0.3) \
            .scale(0.8)

    # -------------------------------------------------
    # 7. Pack everything
    # -------------------------------------------------
    return VGroup(
        axes,
        x_label,
        y_label,
        x_ticks,
        y_ticks,
        points,
        regression_line,
        stats
    )


class DataAnalysisScene2(ZoomedScene):
    def __init__(self, eye_path=None, trials_path=None, **kwargs):
        super().__init__(
            zoom_factor=0.05,
            zoomed_display_height=6,
            zoomed_display_width=12,
            **kwargs
        )
        self.eye_data = None
        self.trials_data = None

        p1 = eye_path or Path(__file__).parent.parent / "data" / "FIXATIONS.csv"
        p2 = trials_path or Path(__file__).parent.parent / "data" / "TRIALS.csv"

        self.load_data(p1, p2)

    def load_data(self, eye_path, trials_path):
        self.eye_data = pd.read_csv(eye_path)
        self.trials_data = pd.read_csv(trials_path)

    def construct(self):
        radius = 0.1

        trials = import_trials_data(self.trials_data)
        all_participants = import_eye_data(self.eye_data)
        target_participants = [all_participants[0]]+[all_participants[-1]]
        # target_participants = all_participants
        all_plots = []
        for (participant, d) in target_participants:
            trials_x = trials[participant]['i']
            trials_y = trials[participant]['y']

            left_x = d['left']['x']
            left_y = d['left']['y']
            right_x = d['right']['x']
            right_y = d['right']['y']
            plot1 = solid_dashed_plot(left_x, right_x, left_y, right_y, y_offset=0)  # we will move it later

            points1 = VGroup(*[
                Dot(plot1[0].c2p(xi, yi), color=RED, radius=radius)
                for xi, yi in zip(trials_x, trials_y)
            ])


            all_plots.append((points1, plot1))

        # 2. Initial layout: stack them horizontally with a small gap
        gap = 24
        for idx, (points, grp) in enumerate(all_plots):
            grp.shift(idx * gap * RIGHT)

        # 3. Animate creation plot after plot, focusing on one plot at a time
        for idx, (points, grp) in enumerate(all_plots):
            axes, x_lbl, y_lbl, x_ticks, y_ticks, solid_line, dashed_line, legend = grp

            plot = VGroup(
                axes,
                x_lbl,
                y_lbl,
                x_ticks,
                y_ticks,
                solid_line,
                dashed_line,
                legend)

            # --- snap zoom on the current plot (instant, no animation) ---
            margin = 1.30
            frame = self.camera.frame
            frame.set(width=plot.width*margin,
                    height=plot.height*margin,
                    center=plot.get_center())

            # 3a. Pan to the current plot (except for the first one)
            self.play(
                frame.animate.move_to(grp.get_center()),
                run_time=0.4
            )

            if idx == 0:
                runtime = 2
                # 2. Fade-in / grow the y-axis first
                self.play(
                    Create(axes.y_axis),
                    FadeIn(y_ticks),
                    Create(y_lbl),
                    run_time=0.5
                )
                self.wait(2)
                # 3. Fade-in / grow the x-axis next
                self.play(
                    Create(axes.x_axis),
                    FadeIn(x_ticks),
                    Create(x_lbl),
                    run_time=0.5
                )
                self.wait(2)
            else:
                runtime = 0.2
                # 3a. Axes, labels, ticks
                self.play(
                    *[Create(mob) for mob in [axes, x_lbl, y_lbl, x_ticks, y_ticks]],
                    run_time=0.4
                )
            # 3b. Solid line + its legend entry
            self.play(
                Create(solid_line),
                FadeIn(legend[0]),
                FadeIn(legend[1]), run_time=runtime)

            self.wait(1)
            # 3c. Dashed line + its legend entry
            self.play(
                Create(dashed_line),
                FadeIn(legend[2]), run_time=runtime)

            self.wait(2)

            x = axes.x_axis.get_center()[0]
            y = axes.y_axis.get_end()[1]
            legend2 = VGroup(
                    Dot(axes.c2p(x, y), color=RED, radius=radius),
                    Text("Acertos", font_size=25))\
                .arrange(RIGHT, buff=0.2)\
                .to_edge(UP).shift(idx * gap * RIGHT)

            points.shift(idx * gap * RIGHT)

            new_y_lbl = Text("Proporção", font_size=28)\
                .next_to(axes.y_axis.get_end(), UP)

            self.play(
                Create(points),
                ReplacementTransform(legend, legend2,
                                     runtime=runtime),
                ReplacementTransform(y_lbl, new_y_lbl,
                                     runtime=runtime)
            )

            self.wait(2)

        # 4.  ———  only on the last plot  ———
        if idx == len(all_plots) - 1:

            # 4a. zoom out so we can see the whole scene
            self.play(
                self.camera.frame.animate.scale(1.3).to_edge(UP),
                run_time=0.4
            )

            # 4b. write formulas
            formula2 = VGroup(
                MathTex(r"y_i="),
                MathTex(r"\frac{\sum \text{acertos}_i}{\sum \text{tentativas}_i}")
                ).arrange(RIGHT)
            formula1 = VGroup(
                MathTex(r"x_i ="),
                VGroup(
                    MathTex(r"\bar{x}_{\substack{\text{olhar palavra}\\ \text{(sílaba nova esquerda)}_i}} -"),
                    MathTex(r"\bar{x}_{\substack{\text{olhar palavra}\\ \text{(sílaba nova direita)}_i}}")
                ).arrange(RIGHT)
            ).arrange(RIGHT)

            formula2.next_to(plot, DOWN, buff=0.5)
            self.play(Write(formula2), run_time=1.5)

            acertos = formula2[1]
            self.play(
                legend2.animate.set_color(YELLOW),
                points.animate.set_color(YELLOW),
                run_time=0.3
            )
            self.play(
                ReplacementTransform(
                    VGroup(legend2, points), acertos),
                run_time=1.2
            )

            formula1.next_to(formula2, DOWN, buff=0.5)
            self.play(Write(formula1), run_time=1.5)

            # 4c. morph the solid line into the left summation symbol
            esquerda = formula1[1][0][0]
            self.play(
                solid_line.animate.set_color(YELLOW),
                run_time=0.3
            )
            self.play(
                ReplacementTransform(solid_line, esquerda),
                run_time=1.2
            )

            # 4d. morph the dashed line into the right summation symbol
            direita = formula1[1][1][0]    # \sum_{\text{sílaba nova direita}}
            self.play(
                dashed_line.animate.set_color(YELLOW),
                run_time=0.3
            )
            self.play(
                ReplacementTransform(dashed_line, direita),
                run_time=1.2
            )

            # 4e. fade the rest of the plot (optional)
            self.play(
                *[FadeOut(m) for m in [axes, x_lbl, y_lbl, new_y_lbl, points,
                                       x_ticks, y_ticks, legend, legend2]],
                run_time=0.5
            )
            self.wait(3)

            # 5. Extend the formula1 with the textual label
            extra = MathTex(r"= \text{Índice de acompanhamento da sílaba nova}")
            extra2 = MathTex(r"= \text{Índice de desempenho}")
            # -------------------------------------------------
            # One-off layout to find the final positions
            # -------------------------------------------------
            tmp = VGroup(
                    VGroup(
                        formula2.copy(), extra2.copy())\
                            .arrange(RIGHT, buff=0.2),
                    VGroup(
                        formula1.copy(), extra.copy())\
                            .arrange(RIGHT, buff=0.2))\
                    .arrange(DOWN, buff=0.5)

            tmp.move_to(self.camera.frame.get_center() + DOWN * 0.5)

            formula2_target = tmp[0][0].get_center()  # keeps current pos
            extra2_target   = tmp[0][1].get_center()     # final pos of extra
            formula1_target = tmp[1][0].get_center()  # keeps current pos
            extra_target   = tmp[1][1].get_center()     # final pos of extra
            scatter_target = tmp.get_center()     # final pos of scatter
            tmp.remove_updater(lambda m: m)   # just to be safe; not strictly needed
            tmp = None                        # discard the temporary group

            # -------------------------------------------------
            # Animate: move formulas first, then write extras
            # -------------------------------------------------
            self.play(formula2.animate.move_to(formula2_target))
            self.play(Write(extra2.move_to(extra2_target)))
            self.play(formula1.animate.move_to(formula1_target))
            self.play(Write(extra.move_to(extra_target)))
            self.wait(1)

        # -------------------------------------------------
        # 6.  Build the correlation scatter for all 22 subjects
        # -------------------------------------------------
        scatter_plot_data = import_scatter_plot_data(self.eye_data)

        scatter = correlation_plot(scatter_plot_data)
        scatter.move_to(scatter_target)

        # -------------------------------------------------
        # 7.  Transition: formula1 ➜ scatter plot
        # -------------------------------------------------

        axes, x_label, y_label, x_ticks, y_ticks, points, regression_line, stats = scatter

        # 7b. dissolve formulas+extras into the axes
        self.play(
            ReplacementTransform(VGroup(formula1, extra), axes.x_axis),
            FadeIn(x_ticks),
            Create(x_label),
            run_time=1.2
        )
        self.wait(2)

        self.play(
            ReplacementTransform(VGroup(formula2, extra2), axes.y_axis),
            FadeIn(y_ticks),
            Create(y_label),
            run_time=1.2
        )
        self.wait(2)

        self.play(
            Create(points),
            run_time=1
        )
        self.wait(2)

        self.play(
            Create(regression_line),
            Create(stats),
            run_time=0.5
        )
        self.wait(4)

        self.play(
            FadeOut(scatter),
            run_time=1.5
        )


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    scene = DataAnalysisScene2()
    scene.render()