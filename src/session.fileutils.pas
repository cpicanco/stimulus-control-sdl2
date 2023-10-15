{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.fileutils;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils;

procedure FindFilesFor(out AStimuliArray: TStringArray;
  AFolder : string;
  AExtensions : string = '*.bmp;*.jpg');

procedure AppendFilesTo(var AStimuliArray: TStringArray;
  AFolder: string;
  AExtensions : string = '*.bmp;*.jpg');

procedure GetDesignFilesFor(AStrings : TStrings);

procedure LoadMessageFromFile(var AMessage : string; AFilename : string);

function NewConfigurationFile : string;
function LoadConfigurationFile(AFilename : string) : string;

implementation

uses
  FileUtil
  , LazFileUtils
  , session.pool
  , session.constants
  , session.configurationfile
  ;

procedure FindFilesFor(out AStimuliArray: TStringArray;
  AFolder: string;
  AExtensions : string = '*.bmp;*.jpg');
var
  Files : TStringList;
  i : integer;
begin
  AStimuliArray := nil;
  Files := TStringList.Create;
  try
    FindAllFiles(Files, AFolder, AExtensions, True);
    SetLength(AStimuliArray, Files.Count);
    for i := Low(AStimuliArray) to High(AStimuliArray) do
      AStimuliArray[i] := Files[i];
  finally
    Files.Free;
  end;
end;

procedure GetDesignFilesFor(AStrings : TStrings);
var
  i : integer;
const
  LDefaultExtension = '*.csv';
  LDefaultFolder    = 'design';
begin
  FindAllFiles(AStrings, LDefaultFolder, LDefaultExtension, False);
  if AStrings.Count > 0 then begin
    for i := 0 to AStrings.Count -1 do begin
      AStrings[i] :=
        ExtractFileNameWithoutExt(ExtractFileNameOnly(AStrings[i]));
    end;
  end;
end;

procedure AppendFilesTo(var AStimuliArray: TStringArray;
  AFolder: string;
  AExtensions : string = '*.bmp;*.jpg');
var
  LOldLength : integer;
  Files : TStringList;
  i : integer;
begin
  LOldLength := Length(AStimuliArray);
  Files := TStringList.Create;
  try
    FindAllFiles(Files, AFolder, AExtensions, True);
    SetLength(AStimuliArray, LOldLength+Files.Count);
    i := Length(AStimuliArray);
    for i := LOldLength to High(AStimuliArray) do
      AStimuliArray[i] := Files[i -LOldLength];
  finally
    Files.Free;
  end;
end;

procedure LoadMessageFromFile(var AMessage : string; AFilename : string);
var
  LStringList : TStringList;
begin
  LStringList := TStringList.Create;
  try
    LStringList.LoadFromFile(AFilename);
    AMessage := LStringList.Text;
  finally
    LStringList.Free;
  end;
end;

function NewConfigurationFile : string;
begin
  //RandSeed := Random(MaxInt);  // Generate a random seed
  RandSeed := 1270036106;
  NewConfigurationFile := Pool.BaseFilePath + 'last_session.ini';
  if FileExists(NewConfigurationFile) then
    DeleteFile(NewConfigurationFile);

  if Assigned(ConfigurationFile) then
    ConfigurationFile.Free;
  ConfigurationFile := TConfigurationFile.Create(NewConfigurationFile);
  ConfigurationFile.CacheUpdates := True;
  ConfigurationFile.WriteInteger(_Main, 'RandSeed', RandSeed);
  ConfigurationFile.Invalidate;
end;

function LoadConfigurationFile(AFilename : string) : string;
begin
  if FileExists(AFilename) then begin
    if Assigned(ConfigurationFile) then
      ConfigurationFile.Free;
    ConfigurationFile := TConfigurationFile.Create(AFilename);
    ConfigurationFile.CacheUpdates := True;
    Result := AFilename;
  end else begin
    raise EFileNotFoundException.Create(AFilename);
  end;
end;


end.

