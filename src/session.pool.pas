{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.pool;

{$mode objfpc}{$H+}

interface

uses
  session.endcriteria, session.counters, session.counters.all;

type

  { TPool }

  TPool = class
    AppName : string;
    RootData : string;
    RootDataResponses: string;
    RootMedia : string;
    BaseFileName : string;
    BaseFilePath : string;
    AssetsBasePath : string;
    ResponsesBasePath : string;
    TimeStart : Extended;
    TestMode : Boolean;
    MonitorToShow : Byte;
    Counters : TCounters;
    Session : TSessionCounters;
    Trial : TTrialCounters;
    Block : TBlockCounters;
    EndCriteria : TEndCriteria;
  end;
var
  Pool : TPool;

implementation

uses SysUtils, SDL2
  , FileUtil
  ;

initialization
  Pool := TPool.Create;
  with Pool do begin
    AppName := 'Stimulus Control';
    BaseFileName := '';
    BaseFilePath := SDL_GetBasePath();
    RootData := BaseFilePath + 'data' + DirectorySeparator;
    RootDataResponses := '';
    RootMedia := BaseFilePath +  'media' + DirectorySeparator;
    AssetsBasePath:='assets'+ DirectorySeparator;
    ResponsesBasePath:= 'responses' + DirectorySeparator;
    ForceDirectories(RootData);
    ForceDirectories(RootMedia);
    ForceDirectories(RootMedia+AssetsBasePath);
    MonitorToShow := 0;
    TestMode := False;
    EndCriteria := nil;
  end

finalization
  Pool.Free;

end.

