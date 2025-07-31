from pathlib import Path

from manim import *
import numpy as np
import pandas as pd

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


def import_data(df):
    data = []
    for participant, cd_data in df.groupby('participant', sort=False):
        sorted_data = cd_data.sort_values(['session', 'trial_in_session'])

        measures = ['proportion', 'frequency']
        grouped_data = {}
        for measure in measures:
            if measure == 'frequency':
                target_measure = [
                        np.cumsum(sorted_data['letter1_count'].values),
                        np.cumsum(sorted_data['letter2_count'].values),
                        np.cumsum(sorted_data['letter3_count'].values),
                        np.cumsum(sorted_data['letter4_count'].values)
                    ]
            elif measure == 'proportion':
                target_measure = [
                    sorted_data['letter1_count'].values,
                    sorted_data['letter2_count'].values,
                    sorted_data['letter3_count'].values,
                    sorted_data['letter4_count'].values
                ]

                # Stack the letter fixation counts into a 2D array
                stacked_counts = np.vstack(target_measure)

                # Calculate the total fixation count for each iteration
                totals = stacked_counts.sum(axis=0)

                # Avoid division by zero by replacing totals of 0 with a small epsilon
                totals = np.where(totals == 0, 1e-8, totals)

                # Compute the proportions and convert to percentages
                percentages = (stacked_counts / totals)

                # Split the proportions back into a list of arrays for each letter
                letters_proportion = [percentages[i] for i in range(len(target_measure))]
                # Calculate moving average window=4 for each letter's proportions
                window_size = 10
                kernel = np.ones(window_size) / window_size  # Kernel [0.25, 0.25, 0.25, 0.25]
                target_measure = []

                for prop in letters_proportion:
                    n = len(prop)
                    if n < window_size:
                        # Not enough trials: return array of NaNs
                        ma = np.full(n, np.nan)
                    else:
                        # Compute moving averages for valid windows (positions 3 to end)
                        valid_ma = np.convolve(prop, kernel, mode='valid')
                        # Initialize array with NaNs
                        ma = np.full(n, np.nan)
                        # Fill valid moving averages starting at index 3
                        ma[window_size-1:] = valid_ma

                    target_measure.append(ma)

            trials = [
                [i+1 for i in range(0, len(m))] for m in target_measure]

            grouped_data[measure] = {
                'x' : trials,
                'y' : target_measure
            }

        data.append((participant, grouped_data))
    return data


def base_plot(x_lines, y_lines, y_offset=0.0):

    masks  = [~np.isnan(y) for y in y_lines]

    x_clean  = [np.array(x)[mask] for x, mask in zip(x_lines, masks)]
    y_clean  = [np.array(y)[mask] for y, mask in zip(y_lines, masks)]

    y_concat = np.concatenate(y_clean)
    # x_concat = np.concatenate(x_clean)

    if max(y_concat) > 1:
        y_maximum = 1100
        y_caption = "Duração\nacumulada"
        y_step = 200
    else:
        y_maximum = 1.01
        y_caption = "Proporção"
        y_step = 0.25

    x_range = [0, 141, 20]
    y_range = [0.0, y_maximum, y_step]

    axes = Axes(
        x_range=x_range,
        y_range=y_range,
        x_length=12,
        y_length=6,
        axis_config={"tip_width": 0.2, "tip_height": 0.2},
        tips=False) \
        .shift(y_offset * DOWN)

    # axis labels
    x_label = Text("Tentativas", font_size=28)\
        .next_to(axes.x_axis.get_end(), RIGHT)

    y_label = Text(y_caption,
                   font_size=28,
                   line_spacing=0.6,
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

    colors = [BLUE, ORANGE, GREEN, PURPLE]

    # ------------------------------------------------------------------
    # solid lines
    # ------------------------------------------------------------------
    solid_lines = [axes.plot(
        lambda t: np.interp(t, x, y),
        color=color,
        stroke_width=4) \
        for x, y, color in zip(x_clean, y_clean, colors)]

    solid_lines = VGroup(*solid_lines)

    # ------------------------------------------------------------------
    # legend
    # ------------------------------------------------------------------
    legend_lines = [
        VGroup(
            Line(ORIGIN, RIGHT*0.5, color=color, stroke_width=4),
            Text(f"{i+1}ª", font_size=25)).arrange(RIGHT, buff=0.2) \
        for i, color in enumerate(colors)
    ]

    legend = VGroup(
        Text("Olhar Letra (Fixações)", font_size=25),
        *legend_lines
    ).arrange(DOWN, aligned_edge=LEFT)\
        .to_edge(UP)\
        .shift(y_offset * DOWN)

    group = VGroup(
        axes,
        x_label,
        y_label,
        x_ticks,
        y_ticks,
        solid_lines,
        legend)
    group.shift(y_offset * DOWN)
    return group


class DataAnalysisScene1(ZoomedScene):
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
        # def pulse_updater(mob, dt):
        #     mob.scale(1 + 0.01 * np.sin(2 * PI * self.time))

        def breathe_updater(mob, dt):
            mob.set_stroke(opacity=0.7 + 0.3 * np.sin(2 * PI * self.time))

        radius = 0.1

        trials = import_trials_data(self.trials_data)
        all_participants = import_data(self.eye_data)
        target_participants = [all_participants[0]]+[all_participants[-1]]
        # target_participants = all_participants
        all_plots = []
        for (participant, d) in target_participants:
            trials_x = trials[participant]['x']
            trials_y = trials[participant]['y']

            x_lines = d['proportion']['x']
            y_lines = d['proportion']['y']
            plot1 = base_plot(x_lines, y_lines, y_offset=0)  # we will move it later

            points1 = VGroup(*[
                Dot(plot1[0].c2p(xi, yi), color=RED, radius=radius)
                for xi, yi in zip(trials_x, trials_y)
            ])

            x_lines = d['frequency']['x']
            y_lines = d['frequency']['y']
            plot2 = base_plot(x_lines, y_lines, y_offset=0)

            all_plots.append((points1, plot1, plot2))

        # 2. Initial layout: stack them horizontally with a small gap
        gap = 20
        for idx, (points, grp, grp2) in enumerate(all_plots):
            grp.shift(idx * gap * RIGHT)
            if idx == 0:
                grp2.shift((idx+1) * (gap/2) * DOWN)
            else:
                grp2.shift(idx * (gap) * RIGHT)
                grp2.shift(idx * (gap/2) * DOWN)

        # 3. Animate creation plot after plot, focusing on one plot at a time
        for idx, (points, grp, _) in enumerate(all_plots):
            axes, x_lbl, y_lbl, x_ticks, y_ticks, solid_lines, legend = grp
            # if idx > 0:
            y_axis_proportion = axes.y_axis.copy()
            # y_ticks_right = y_ticks.copy()

            plot = VGroup(
                axes,
                x_lbl,
                y_lbl,
                x_ticks,
                y_ticks,
                solid_lines,
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
                runtime = 1
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
                    run_time=0.25
                )
            # 3b. Solid line + its legend entry
            for n, line in enumerate(solid_lines):
                anim = []
                anim.append(Create(line))
                if n == 0:
                    anim.append(FadeIn(legend[0]))
                anim.append(FadeIn(legend[n+1]))

                self.play(*anim, run_time=runtime)
                self.wait(1)

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

            lines_to_emphasize1 = [1, 2]
            lines_to_emphasize2 = [0, 3]

            for n in lines_to_emphasize2:
                solid_lines[n].set_stroke(opacity=0.3)

            for n in lines_to_emphasize1:
                solid_lines[n].add_updater(breathe_updater)

            self.wait(2)

            for n in lines_to_emphasize2:
                solid_lines[n].set_stroke(opacity=1.0)

            for n in lines_to_emphasize1:
                solid_lines[n].remove_updater(breathe_updater)
            
            self.wait(2)

        ######
        # Second plot
        #####



        for idx, (points, _, grp) in enumerate(all_plots):
            axes, x_lbl, y_lbl, x_ticks, y_ticks, solid_lines, legend = grp

            plot = VGroup(
                axes,
                x_lbl,
                y_lbl,
                x_ticks,
                y_ticks,
                solid_lines,
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
                runtime = 0.2
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
            for n, line in enumerate(solid_lines):
                anim = []
                anim.append(Create(line))
                if n == 0:
                    anim.append(FadeIn(legend[0]))
                anim.append(FadeIn(legend[n+1]))

                self.play(*anim, run_time=runtime)
                self.wait(1)


            if idx == 0:
                position = (idx+1) * (gap/2) * DOWN
            else:
                position = idx * (gap/2) * DOWN

        #     x = axes.x_axis.get_center()[0]
        #     y = axes.y_axis.get_end()[1]
        #     legend2 = VGroup(
        #             Dot(axes.c2p(x, y), color=RED, radius=radius),
        #             Text("Acertos", font_size=25))\
        #         .arrange(RIGHT, buff=0.2)\
        #         .to_edge(UP).shift(idx * gap * RIGHT)

            points.shift(position)

            y_axis_proportion.move_to(
                axes.x_axis.get_right(),   # right end of the x-axis
                aligned_edge=DOWN          # align the y-axis’s own origin (bottom) to that point
            )

            y_ticks_right = VGroup(*[
                Text(f"{i:.1f}", font_size=20).next_to(axes.c2p(0, i), RIGHT, buff=0.2)
                for i in np.arange(0, 1.01, 0.25)
            ])

            new_y_lbl = Text("Proporção", font_size=28)\
                .next_to(y_axis_proportion.get_end(), UP)

            self.play(
                Create(points),
                Create(y_axis_proportion),
                FadeIn(y_ticks_right),
                Create(new_y_lbl)
            )

            self.wait(2)

            lines_to_emphasize1 = [1, 2]
            lines_to_emphasize2 = [0, 3]

            for n in lines_to_emphasize2:
                solid_lines[n].set_stroke(opacity=0.3)

            for n in lines_to_emphasize1:
                solid_lines[n].add_updater(breathe_updater)

            self.wait(2)

            for n in lines_to_emphasize2:
                solid_lines[n].set_stroke(opacity=1.0)

            for n in lines_to_emphasize1:
                solid_lines[n].remove_updater(breathe_updater)


        # 4a. zoom out so we can see the whole scene
        self.play(
            self.camera.frame.animate.scale(2.0).to_edge(LEFT),
            run_time=0.4
        )

        self.wait(4)






if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    scene = DataAnalysisScene1()
    scene.render()