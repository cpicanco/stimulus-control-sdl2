
from syllables_permutations import PermutationScene
from data_analysis_scene1 import DataAnalysisScene1
from data_analysis_scene2 import DataAnalysisScene2

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    for scene in [PermutationScene(), DataAnalysisScene1(), DataAnalysisScene2()]:
        scene.render()