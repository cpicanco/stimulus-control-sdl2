{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.loggers;

{$mode ObjFPC}{$H+}

interface

uses
  Classes;

type

  { TLogger }

  TLogger = class
    private
      FFilename : string;
      FFileIndex : integer;
      FTextFile : TextFile;
    public
      constructor Create(AFilename : string);
      destructor Destroy; override;
      class function Row(Cols: array of string;
        ALineEnding : string = LineEnding): string;
      class function GetBaseFilename: string; static;
      class procedure SetHeader(ASessionName: string; AParticipantName: string); static;
      class procedure SetFooter; static;
      procedure SaveData(S: string);
      procedure SaveLine(S: string);
      property Filename : string read FFilename;
      property FileIndex : integer read FFileIndex;
  end;


implementation

uses SysUtils
  , LazFileUtils
  , session.pool
  , session.loggers.instances;

const FirstBasename: string = '001';

{ TLogger }

constructor TLogger.Create(AFilename: string);
  function FilenameNoOverride(S: string): string;
  var
    i : Integer;
    FilePath, LExtension: string;
  begin
    if S.IsEmpty then
      raise Exception.Create('TLogger.Create: Filename cannot be empty.');
    ForceDirectoriesUTF8(ExtractFilePath(S));
    FilePath := ExtractFilePath(S);
    LExtension := ExtractFileExt(S);
    i := 0;

    // ensure to never override an exinting file
    S := AFilename;
    while FileExistsUTF8(S) do begin
      Inc(i);
      S := FilePath +
           StringOfChar(#48, 3 - Length(IntToStr(i))) + IntToStr(i) +
           LExtension;
    end;
    FFileIndex := i;
    Result := S;
  end;
begin
  FFilename := FilenameNoOverride(AFilename);
  AssignFile(FTextFile, FFilename);
  System.Rewrite(FTextFile);
end;

destructor TLogger.Destroy;
begin
  System.Close(FTextFile);
  inherited Destroy;
end;

class function TLogger.GetBaseFilename: string;
begin
  if DataFilename = '' then
    Result := FirstBasename
  else
    Result := ExtractFileNameWithoutExt(DataFilename);
end;

class function TLogger.Row(Cols: array of string; ALineEnding: string): string;
const
  TAB = #9;
var
  i : Integer;
  LastColumn : Integer;
begin
  Result := '';
  LastColumn := High(Cols);
  for i := 0 to LastColumn do
    if i < LastColumn then
      Result := Result + Cols[i]+TAB
    else
      Result := Result + Cols[i]+ALineEnding;
end;

class procedure TLogger.SetHeader(ASessionName: string;
  AParticipantName: string);
var
  LHeader : string;
  LFirstFilename : string;
begin
  LFirstFilename := Pool.BaseFileName + FirstBasename;
  LHeader :=
    HSUBJECT_NAME + #9 + AParticipantName + LineEnding +
    HSESSION_NAME + #9 + ASessionName + LineEnding +
    HBEGIN_TIME + #9 + DateTimeToStr(Date) + #9 + TimeToStr(Time) + LineEnding;
  DataFilename := CreateLogger(LGData, LFirstFilename, LHeader);
  TimestampsFilename := CreateLogger(LGTimestamps, LFirstFilename, LHeader);
  Pool.BaseFilename := GetBaseFilename;
  Pool.Session.ID := ExtractFileNameOnly(Pool.BaseFilename).ToInteger;
end;

class procedure TLogger.SetFooter;
var
  Footer : string;
begin
  Footer := HEND_TIME + #9 + DateTimeToStr(Date) + #9 + TimeToStr(Time)+ LineEnding;
  FreeLogger(LGTimestamps, Footer);
  FreeLogger(LGData, Footer);
end;

procedure TLogger.SaveData(S: string);
begin
  Write(FTextFile, S);
  System.Flush(FTextFile);
end;

procedure TLogger.SaveLine(S: string);
begin
  WriteLn(FTextFile, S);
  System.Flush(FTextFile);
end;



end.

