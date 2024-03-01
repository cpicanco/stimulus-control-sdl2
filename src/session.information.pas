unit session.information;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, SDL2, sdl.app.grids.types;

type

  TInformation = record
    Basename : string;
    ParticipantName : string;
    SessionName : string;
    SessionResult : string;
    Grid : TMatrix;
    Monitor : TSDL_Rect;
    SessionStart : TDateTime;
    SessionEnd : TDateTime;
    SessionDuration : TDateTime;
  end;

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
  Result.Basename := ExtractFileNameWithoutExt(AFileName);
  LExtension := ExtractFileExt(AFileName);
  if LExtension <> GExtention then begin
    LFilename := Result.Basename+GExtention;
  end else begin
    LFilename := AFileName;
  end;

  LInfoFile := TStringList.Create;
  try
    LInfoFile.LoadFromFile(LFilename);
    LInfoFile.NameValueSeparator := GSeparator;

    with Result, LInfoFile do begin
      ParticipantName := Trim(Values[HSUBJECT_NAME]);
      SessionName     := Trim(Values[HSESSION_NAME]);
      SessionResult   := Trim(Values[HSESSION_RESULT]);
      //Grid :=            Trim(MatrixFromJSON(LInfoFile.Values[HGRID]));
      //Monitor :=         Trim(MonitorFromJSON(Values[HMONITOR]));
      //SessionStart :=    Trim(StrToDateTime(Values[HBEGIN_TIME]));
      //SessionEnd :=      Trim(StrToDateTime(Values[HEND_TIME]));
      //SessionDuration := Trim(StrToDateTime(Values[HDURATION]));
    end;
  finally
    LInfoFile.Free;
  end;
end;

procedure SetSessionResult(AResult: string);
begin
  SessionResult := AResult;
end;


end.
