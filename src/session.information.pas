{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.information;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, session.loggers.types;

  function LoadInformationFromFile(const AFileName: string): TInformation;
  procedure SetSessionResult(AResult : string);

implementation

uses LazFileUtils, session.loggers.writerow.information;

function LoadInformationFromFile(const AFileName: string): TInformation;
var
  LInfoFile: TStringList;
  LExtension : string;
  LFilename : string;
begin
  if FileExists(AFileName) then begin
    Result.Basename := ExtractFileNameWithoutExt(AFileName);
    LExtension := ExtractFileExt(AFileName);
    if LExtension <> GExtention then begin
      LFilename := Result.Basename+GExtention;
    end else begin
      LFilename := AFileName;
    end;

    LInfoFile := TStringList.Create;
    try
      LInfoFile.NameValueSeparator := GSeparator;
      LInfoFile.LoadFromFile(LFilename);
      LFilename := LInfoFile.Text;
      with Result, LInfoFile do begin
        if Trim(Values[HVERSION]).IsEmpty then begin
          Version := 0;
        end else begin
          Version := Byte(Trim(Values[HVERSION]).ToInteger);
        end;
        ParticipantName := Trim(Values[HSUBJECT_NAME]);
        SessionName     := Trim(Values[HSESSION_NAME]);
        SessionResult   := Trim(Values[HSESSION_RESULT]);
        SessionDesignFolder := Trim(Values[HSESSION_DESIGN]);
        //Grid :=            Trim(MatrixFromJSON(LInfoFile.Values[HGRID]));
        //Monitor :=         Trim(MonitorFromJSON(Values[HMONITOR]));
        //SessionStart :=    Trim(StrToDateTime(Values[HBEGIN_TIME]));
        //SessionEnd :=      Trim(StrToDateTime(Values[HEND_TIME]));
        //SessionDuration := Trim(StrToDateTime(Values[HDURATION]));
      end;
    finally
      LInfoFile.Free;
    end;
  end else begin
    Result.Basename := '';
  end;
end;

procedure SetSessionResult(AResult: string);
begin
  SessionResult := AResult;
end;


end.

