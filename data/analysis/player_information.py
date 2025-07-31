
from player_utils import as_dict

def load_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"An error occurred while loading the DataFrame: {e}")

class GazeInfo:
    __info_extension__ = '.info.processed'
    __gaze_info_extension__ = '.gaze.info'

    __HeaderVersion__ = 'Version'
    __ParticipantName__ = 'Nome_do_sujeito'
    __SessionName__ = 'Nome_da_sessao'
    __SessionResult__ = 'Resultado'
    __Grid__ = 'Grade_de_estimulos'
    __StartDate__ = 'Data_Inicio'
    __StartTime__ = 'Hora_Inicio'
    __EndDate__ = 'Data_Termino'
    __EndTime__ = 'Hora_Termino'
    __Duration__ = 'Duration'

    __ClientVersion__ = 'ClientVersion'
    __TimeTickFrequency__ = 'TimeTickFrequency'
    __CameraWidth__ = 'CameraWidth'
    __CameraHeight__ = 'CameraHeight'
    __ProductID__ = 'ProductID'
    __ProductBus__ = 'ProductBus'
    __ProductRate__ = 'ProductRate'
    __SerialID__ = 'SerialID'
    __CompanyID__ = 'CompanyID'
    __APIID__ = 'APIID'
    __Monitor__ = 'Monitor'
    __ScreenWidth__ = 'ScreenWidth'
    __ScreenHeight__ = 'ScreenHeight'

    def __init__(self, base_filename):
        self.base_filename = base_filename
        self.HeaderVersion = None
        self.ParticipantName = None
        self.SessionName = None
        self.SessionResult = None
        self.Grid = None
        self.StartDate = None
        self.StartTime = None
        self.EndDate = None
        self.EndTime = None
        self.Duration = None

        self.ClientVersion = None
        self.TimeTickFrequency = None
        self.CameraWidth = None
        self.CameraHeight = None
        self.ProductID = None
        self.ProductBus = None
        self.ProductRate = None
        self.SerialID = None
        self.CompanyID = None
        self.APIID = None
        self.ScreenWidth = None
        self.ScreenHeight = None
        self.load_from_file()

    def load_from_file(self):
        info_filename = self.base_filename + self.__info_extension__
        gaze_info_filename = self.base_filename + self.__gaze_info_extension__

        lines = load_from_file(info_filename)
        lines_gaze = load_from_file(gaze_info_filename)

        if lines_gaze is not None:
            lines += load_from_file(gaze_info_filename)

        for line in lines:
            def value(delimiter=':'):
                return line.split(delimiter)[1].strip()

            if line.startswith(self.__ClientVersion__):
                self.ClientVersion = int(value())
            elif line.startswith(self.__TimeTickFrequency__):
                self.TimeTickFrequency = int(value())
            elif line.startswith(self.__CameraWidth__):
                self.CameraWidth = int(value())
            elif line.startswith(self.__CameraHeight__):
                self.CameraHeight = int(value())
            elif line.startswith(self.__ProductID__):
                self.ProductID = value()
            elif line.startswith(self.__ProductBus__):
                self.ProductBus = value()
            elif line.startswith(self.__ProductRate__):
                self.ProductRate = int(value())
            elif line.startswith(self.__SerialID__):
                self.SerialID = value()
            elif line.startswith(self.__CompanyID__):
                self.CompanyID = value()
            elif line.startswith(self.__APIID__):
                self.APIID = float(value())
            elif line.startswith(self.__Monitor__):
                monitor = value(self.__Monitor__+':')
                monitor = as_dict(monitor)
                # extract screen width and height
                self.ScreenWidth = int(monitor['w'])
                self.ScreenHeight = int(monitor['h'])
            elif line.startswith(self.__HeaderVersion__):
                self.HeaderVersion = int(value())
            elif line.startswith(self.__ParticipantName__):
                self.ParticipantName = value()
            elif line.startswith(self.__SessionName__):
                self.SessionName = value()
                self.Cycle = self.SessionName.split('-')[0].replace('Ciclo', '')
                self.Condition = self.SessionName.split('-')[1]
            elif line.startswith(self.__SessionResult__):
                self.SessionResult = value()
            elif line.startswith(self.__Grid__):
                grid = value(self.__Grid__+':')
                self.Grid = as_dict(grid)
            elif line.startswith(self.__StartDate__):
                self.StartDate = value()
            elif line.startswith(self.__StartTime__):
                self.StartTime = value()
            elif line.startswith(self.__EndDate__):
                self.EndDate = value()
            elif line.startswith(self.__EndTime__):
                self.EndTime = value()
            elif line.startswith(self.__Duration__):
                self.Duration = float(value())


    @property
    def relations(self):
        name = self.SessionName.split('-')[2:]
        name = '-'.join(name)
        events = self.target_trial_events()
        return events[name].keys()

    def relation_stimuli(self, relation : str):
        name = self.SessionName.split('-')[2:]
        name = '-'.join(name)
        events = self.target_trial_events()
        return events[name][relation].keys()

    def target_trial_events(self):
        """
        Target trial stimuli/events used to filter data inside a trial.
            samp = Samples
            comp = Comparisons
            begin = start of event
            end = end of event
        """
        AB = {'samp': {'begin': 'MTSStimuli.ModalityAB.Show', 'end': 'Comparisons.Start'},
              'comp': {'begin': 'Comparisons.Start', 'end': 'MTSStimuli.ModalityAB.Hide'}}
        AC = {'samp': {'begin': 'MTSStimuli.ModalityAC.Show', 'end': 'Comparisons.Start'},
              'comp': {'begin': 'Comparisons.Start', 'end': 'MTSStimuli.ModalityAC.Hide'}}
        CB = {'samp': {'begin': 'MTSStimuli.ModalityCB.Show', 'end': 'Comparisons.Start'},
              'comp': {'begin': 'Comparisons.Start', 'end': 'MTSStimuli.ModalityCB.Hide'}}
        BC = {'samp': {'begin': 'MTSStimuli.ModalityBC.Show', 'end': 'Comparisons.Start'},
              'comp': {'begin': 'Comparisons.Start', 'end': 'MTSStimuli.ModalityBC.Hide'}}
        BB = {'samp': {'begin': 'MTSStimuli.ModalityBB.Show', 'end': 'Comparisons.Start'},
              'comp': {'begin': 'Comparisons.Start', 'end': 'MTSStimuli.ModalityBB.Hide'}}
        CD = {'samp': {'begin': 'MTSStimuli.ModalityCD.Show', 'end': 'MTSStimuli.ModalityCD.Hide'}}

        return {
            'Pre-treino':
                {'BB': BB, 'CD': CD},
            'Sondas-CD-Palavras-12-ensino-8-generalizacao':
                {'CD': CD},
            'Treino-AB':
                {'AB': AB, 'CD': CD},
            'Treino-AC-CD':
                {'AC': AC, 'CD': CD},
            'Treino-AC-Ref-Intermitente':
                {'AC': AC},
            'Sondas-BC-CB-Palavras-de-ensino':
                {'BC': BC, 'CB': CB},
            'Sondas-BC-CB-Palavras-reservadas':
                {'BC': BC, 'CB': CB},
            'Sondas-CD-Palavras-generalizacao-reservadas':
                {'CD': CD},
            'Sondas-AC-Palavras-generalizacao-reservadas':
                {'AC': AC},
            'Sondas-CD-Palavras-30-Todas':
                {'CD': CD},
        }

if __name__ == '__main__':
    from fileutils import cd, data_dir

    data_dir()
    cd('1-JOP')
    cd('analysis')
    cd('2024-07-22')
    cd('0')
    info = GazeInfo('000')
    print(f'ScreenSize: {info.ScreenWidth}x{info.ScreenHeight}')
    print(f'Grid: {info.Grid}')
    print(f'ClientVersion: {info.ClientVersion}')
    print(f'APIID: {info.APIID}')
    print(f'CompanyID: {info.CompanyID}')
    print(f'ProductID: {info.ProductID}')
    print(f'ProductRate: {info.ProductRate}')
    print(f'ProductBus: {info.ProductBus}')
    print(f'SerialID: {info.SerialID}')
    print(f'CameraWidth: {info.CameraWidth}')
    print(f'CameraHeight: {info.CameraHeight}')
    print(f'HeaderVersion: {info.HeaderVersion}')
    print(f'ParticipantName: {info.ParticipantName}')
    print(f'SessionName: {info.SessionName}')
    print(f'SessionResult: {info.SessionResult}')
    print(f'StartDate: {info.StartDate}')
    print(f'StartTime: {info.StartTime}')
    print(f'EndDate: {info.EndDate}')
    print(f'EndTime: {info.EndTime}')
    print(f'TimeTickFrequency: {info.TimeTickFrequency}')
    print(f'Duration: {info.Duration}')
    data_dir()