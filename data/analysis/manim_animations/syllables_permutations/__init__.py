from manim import *

class PermutationScene(Scene):
    def construct(self):
        TEACHING_COLOR = GREEN
        NEW_COLOR = YELLOW

        ni_text = "ni"
        bo_text = "bo"
        fa_text = "fa"
        le_text = "le"

        fontname = 'Picanco et al'
        fontsize = 67

        ni_ = Text(ni_text, font=fontname, font_size=fontsize)
        bo_ = Text(bo_text, font=fontname, font_size=fontsize)
        fa_ = Text(fa_text, font=fontname, font_size=fontsize)
        le_ = Text(le_text, font=fontname, font_size=fontsize)

        syllables_group_symbols = VGroup(
            ni_, bo_, fa_, le_
        ).arrange(RIGHT, buff=1.0)

        linear_table = VGroup(
            Text("Sílabas de ensino", font_size=36).move_to(UP),
            syllables_group_symbols
        ).arrange(DOWN, buff=1.0)

        self.play(Write(linear_table))
        self.wait(4)

        ni = Text(ni_text, color=TEACHING_COLOR)
        bo = Text(bo_text, color=TEACHING_COLOR)
        fa = Text(fa_text, color=TEACHING_COLOR)
        le = Text(le_text, color=TEACHING_COLOR)

        syllables_group_linear = VGroup(
            ni, bo, fa, le
        ).arrange(RIGHT, buff=0.8)

        for s1, s2 in zip(linear_table[1], syllables_group_linear):
            self.play(Transform(s1, s2), run_time=0.5)

        self.wait(1)

        # --------  existing cycles  ----------
        ni = Text(ni_text, color=TEACHING_COLOR)
        bo = Text(bo_text, color=TEACHING_COLOR)
        fa = Text(fa_text, color=TEACHING_COLOR)
        le = Text(le_text, color=TEACHING_COLOR)

        syllables_group = VGroup(
            VGroup(ni, bo).arrange(RIGHT),
            VGroup(fa, le).arrange(RIGHT),
        ).arrange(DOWN)

        table = VGroup(
            Text("Ciclo 1", font_size=36),
            syllables_group
        ).arrange(DOWN)

        p1 = syllables_group[0][0].get_center()
        p2 = syllables_group[0][1].get_center()
        p3 = syllables_group[1][0].get_center()
        p4 = syllables_group[1][1].get_center()

        positions_by_cycle = [
            [(p1, p2), (p3, p4)],
            [(p2, p3), (p4, p1)],
            [(p4, p2), (p3, p1)],
            [(p2, p1), (p4, p3)],
            [(p3, p2), (p1, p4)],
            [(p2, p4), (p1, p3)]
        ]

        words_text = Text("2 palavras", font_size=36).next_to(syllables_group, LEFT * 1.5)

        for cycle, syllables_positions in enumerate(positions_by_cycle):
            animations = []
            if cycle == 0:
                self.remove(linear_table[0])
                for syllables, positions in zip(syllables_group, syllables_positions):
                    syllables[0].move_to(positions[0])
                    syllables[1].move_to(positions[1])
                animations.extend([FadeIn(words_text), FadeIn(table[0]),
                                   ReplacementTransform(linear_table[1], table[1])])
            else:
                new_words_text = Text(f"{(cycle + 1)*2} palavras", font_size=36).next_to(syllables_group, LEFT * 1.5)
                new_title = Text(f"Ciclo {cycle + 1}", font_size=36)
                new_title.move_to(table[0])
                new_words_text.move_to(words_text)
                animations.append(Transform(table[0], new_title))
                animations.append(ReplacementTransform(words_text, new_words_text))
                words_text = new_words_text
                for syllables, positions in zip(syllables_group, syllables_positions):
                    animations.append(syllables[0].animate.move_to(positions[0]))
                    animations.append(syllables[1].animate.move_to(positions[1]))
            self.play(*animations)
            if (cycle == 0) or (cycle == len(positions_by_cycle) - 1):
                wait_time = 1
            else:
                wait_time = 0.3
            self.wait(wait_time)

        self.remove(words_text)
        self.remove(new_words_text)

        new_title = Text(
            "12 palavras com\nsílabas de ensino",
            font_size=36,
            line_spacing=0.8)
        new_title.move_to(table[0]).shift(UP * 0.5)
        self.play(Transform(table[0], new_title))

        # ---------- 1 ----------  move teaching syllables to a vertical column on the center
        right_col = VGroup(*[Text(s, color=TEACHING_COLOR) for s in ["ni", "bo", "fa", "le"]])
        right_col.arrange(DOWN, buff=0.5)

        self.play(
            table[0].animate.next_to(right_col, UP),
            *[Transform(syllables_group[i // 2][i % 2], right_col[i]) for i in range(4)]
        )
        self.play(
            table.animate.to_edge(UL),
        )

        # ---------- 3 ----------  build the two new tables one after the other
        def create_new_table(words, title_text, teaching_syllables):
            table = VGroup()
            for word in words:
                group = VGroup()
                for syllable in word:
                    color = TEACHING_COLOR if syllable in teaching_syllables else NEW_COLOR
                    group.add(Text(syllable, color=color))
                group.arrange(RIGHT, buff=0.2)
                table.add(group)
            table.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
            title = Text(title_text, font_size=30)
            return VGroup(title, table).arrange(DOWN, buff=0.5)

        new_left_syllables = ["la", "fe", "bi", "no"]
        new_right_syllables = ["lo", "na", "be", "fi"]
        new_left_words = [("la", "ni"), ("fe", "bo"), ("bi", "fa"), ("no", "le")]
        new_right_words = [("ni", "lo"), ("bo", "na"), ("fa", "be"), ("le", "fi")]
        teaching_syllables_list = ["ni", "bo", "fa", "le"]

        # Top Center Left Title
        plus_sign = Text("+", font_size=36).next_to(table[0], RIGHT)
        new_words_title = VGroup(
            Text("8 com", font_size=36),
            Text("sílabas novas", font_size=36, color=NEW_COLOR)
        ).arrange(RIGHT)
        new_words_title.next_to(plus_sign, RIGHT)
        self.play(Write(plus_sign), Write(new_words_title))

        # ---------- 2 ----------  write the new left syllables in a vertical column on the LEFT
        left_col = VGroup(*[Text(s, color=NEW_COLOR) for s in new_left_syllables])
        left_col.arrange(DOWN, buff=0.5).shift(DOWN * 0.5)

        self.play(
            LaggedStart(*(Write(t) for t in left_col), lag_ratio=0.2))

        left_table = create_new_table(new_left_words, 'à esquerda', teaching_syllables_list)

        self.play(LaggedStart(*(Transform(left_col[i], left_table[1][i]) for i in range(4)),
                              lag_ratio=0.2),
                Write(left_table[0]))
        # self.play()

        plus_sign2 = Text("+", font_size=36).next_to(left_table, RIGHT)
        # RIGHT TABLE (words with new syllables on the RIGHT)
        right_table = create_new_table(new_right_words, 'à direita', teaching_syllables_list)
        right_table.next_to(left_table, RIGHT, buff=1.5)
        # first write the new right syllables
        new_right_col = VGroup(*[Text(s, color=NEW_COLOR) for s in new_right_syllables])
        new_right_col.arrange(DOWN, buff=0.5).next_to(left_table, RIGHT, buff=0.5).shift(RIGHT*1.5).shift(DOWN * 0.5)
        self.play(LaggedStart(
            *(Write(t) for t in new_right_col), lag_ratio=0.2),
            Write(plus_sign2))
        # then morph them into the right table
        self.play(
            LaggedStart(*(Transform(new_right_col[i], right_table[1][i]) for i in range(4)),
                              lag_ratio=0.2),
                Write(right_table[0]))
        # self.play()


        self.wait(5)



if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    scene = PermutationScene()
    scene.render()