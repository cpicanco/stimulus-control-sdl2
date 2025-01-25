{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.pool;

{$mode objfpc}{$H+}

interface

uses
  sdl.app
  , session.endcriteria
  , session.counters
  , session.counters.all;

type

  { TPool }

  TPool = record
    App : TSDLApplication;
    Counters : TCounters;
    Session : TSessionCounters;
    Trial : TTrialCounters;
    Block : TBlockCounters;
    EndCriteria : TEndCriteria;
    AppName : string;
    AssetsRootBasePath : string;
    AudioBasePath : string;
    AudioRootBasePath : string;
    BaseDataPath : string;
    BaseFileName : string;
    BasePath : string;
    ConfigurationsRootBasePath: string;
    ConfigurationFilename : string;
    DataResponsesBasePath: string;
    DataRootBasePath : string;
    DesignBasePath : string;
    DesignRootBasePath : string;
    FontsRootBasePath : string;
    ImageBasePath : string;
    ImageRootBasePath : string;
    MediaRootBasePath : string;
    ResponsesBasePath : string;
    RootAudio : string;
    ParticipantName: string;
    SessionName: string;
    MonitorToShow : Byte;
    TestMode : Boolean;
  end;

var
  Pool : TPool;

implementation

uses SysUtils, SDL2, session.strutils;

initialization
  with Pool do begin
    AppName := 'Stimulus Control';

    BasePath := SDL_GetBasePath();

    // data
    DataRootBasePath   := ConcatPaths([BasePath, AsPath('data')]);
    ForceDirectories(DataRootBasePath);

    // config
    ConfigurationsRootBasePath := ConcatPaths([BasePath, AsPath('config')]);
    ForceDirectories(ConfigurationsRootBasePath);

    // data/{Participant}
    BaseDataPath := '';

    // {Filename} 000
    BaseFileName := '';

    // data/{Participant}/{Filename}.ini
    ConfigurationFilename := '';

    // data/{Participant}/responses/
    DataResponsesBasePath := '';
    ResponsesBasePath:= AsPath('responses');

    // design/
    DesignRootBasePath := ConcatPaths([BasePath, AsPath('design')]);
    ForceDirectories(DesignRootBasePath);
    DesignBasePath := '';

    // media/
    MediaRootBasePath := ConcatPaths([BasePath, AsPath('media')]);
    ForceDirectories(MediaRootBasePath);

    // media/fonts/
    FontsRootBasePath := ConcatPaths([MediaRootBasePath, AsPath('fonts')]);
    ForceDirectories(FontsRootBasePath);

    // media/assets/
    AssetsRootBasePath := ConcatPaths([MediaRootBasePath, AsPath('assets')]);
    ForceDirectories(AssetsRootBasePath);

    // media/png/
    ImageRootBasePath := ConcatPaths([MediaRootBasePath, AsPath('png')]);
    ForceDirectories(ImageRootBasePath);

    // media/png/{Participant}/
    ImageBasePath := '';

    // media/wav/
    AudioRootBasePath := ConcatPaths([MediaRootBasePath, AsPath('wav')]);
    ForceDirectories(AudioRootBasePath);

    // media/wav/{CustomFolder}/
    AudioBasePath := '';

    MonitorToShow := 0;
    TestMode := False;
    EndCriteria := nil;
  end

end.

