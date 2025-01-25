{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

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
      class procedure SetHeader; static;
      class procedure SetFooter; static;
      procedure SaveData(S: string);
      procedure SaveLine(S: string);
      property Filename : string read FFilename;
      property FileIndex : integer read FFileIndex;
  end;

implementation

uses SysUtils
  , DateUtils
  , LazFileUtils
  , session.fileutils
  , session.pool
  , session.loggers.types
  , session.loggers.instances;

const FirstBasename: string = '000';

{ TLogger }

constructor TLogger.Create(AFilename: string);
begin
  FFilename := FilenameNoOverride(AFilename, FFileIndex);
  AssignFile(FTextFile, FFilename);
  System.Rewrite(FTextFile);
end;

destructor TLogger.Destroy;
begin
  System.Close(FTextFile);
  inherited Destroy;
end;

class function TLogger.GetBaseFilename: string;
var
  LFilename : string;
begin
  LFilename := InformationFilename;
  if LFilename.IsEmpty then begin
    Result := FirstBasename;
  end else begin
    Result := ExtractFileNameOnly(LFilename);
  end;
end;

class function TLogger.Row(Cols: array of string; ALineEnding: string): string;
const
  LTAB = #9;
var
  i : Integer;
  LastColumn : Integer;
begin
  Result := '';
  LastColumn := High(Cols);
  for i := 0 to LastColumn-1 do begin
    Result := Result + Cols[i]+LTAB
  end;
  Result := Result + Cols[LastColumn]+ALineEnding;
end;

class procedure TLogger.SetHeader;
var
  LFirstFilename : string;
begin
  LFirstFilename := Pool.BaseDataPath + FirstBasename;

  InformationFilename := CreateLogger(LGInfo, LFirstFilename);
  DataFilename := CreateLogger(LGData, LFirstFilename);
  TimestampsFilename := CreateLogger(LGTimestamps, LFirstFilename);

  Pool.BaseFilename := GetBaseFilename;
  Pool.Session.ID := Pool.BaseFilename.ToInteger;
end;

class procedure TLogger.SetFooter;
begin
  FreeLogger(LGTimestamps);
  FreeLogger(LGData);
  FreeLogger(LGInfo);
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

