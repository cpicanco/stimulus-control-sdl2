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


if __name__ == '__main__':
    from fileutils import cd, data_dir

    data_dir()
    cd('1-JOP')
    cd('analysis')
    cd('2024-07-22')
    cd('0')
    info = GazeInfo('000')
    print(f'ScreenSize: {info.ScreenWidth}x{info.ScreenHeight}')
    data_dir()