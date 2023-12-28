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

procedure GetAudioFoldersFor(AStrings : TStrings);
procedure GetAudioFilesFor(AStrings : TStrings);
procedure GetDesignFoldersFor(AStrings : TStrings);
procedure GetDesignFilesFor(AStrings : TStrings);
procedure GetFontFilesFor(AStrings : TStrings);

procedure FreeConfigurationFile;
procedure LoadMessageFromFile(var AMessage : string; AFilename : string);


function NewConfigurationFile : string;
function LoadConfigurationFile(AFilename : string) : string;

implementation

uses
  FileUtil
  , LazFileUtils
  , session.pool
  , session.strutils
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
    Files.Clear;
    Files.Free;
  end;
end;

procedure GetAudioFoldersFor(AStrings : TStrings);
var
  i : integer;
  LDefaultFolder : string;
const
  LFolder = 'media';
  LSubfolder = 'wav';
begin
  LDefaultFolder := ConcatPaths([LFolder, LSubfolder]);
  FindAllDirectories(AStrings, LDefaultFolder, False);
  for i := 0 to AStrings.Count - 1 do begin
    AStrings[i] :=
      AsPath(ExtractFileNameOnly(AStrings[i]));
  end;
end;

procedure GetAudioFilesFor(AStrings: TStrings);
var
  i : integer;
  LDefaultFolder : string;
const
  LDefaultExtension = '*.wav';
  LFolder = 'media';
  LSubfolder = 'assets';
begin
  LDefaultFolder := ConcatPaths([LFolder, LSubfolder]);
  FindAllFiles(AStrings, LDefaultFolder, LDefaultExtension, False);
  for i := 0 to AStrings.Count - 1 do begin
    AStrings[i] :=
      ExtractFileNameWithoutExt(ExtractFileNameOnly(AStrings[i]));
  end;
end;

procedure GetDesignFoldersFor(AStrings: TStrings);
var
  i : integer;
const
  LFolder = 'design';
begin
  FindAllDirectories(AStrings, LFolder, False);
  for i := 0 to AStrings.Count - 1 do begin
    AStrings[i] :=
      AsPath(AStrings[i].Replace(LFolder+DirectorySeparator, ''));
  end;
end;

procedure GetDesignFilesFor(AStrings : TStrings);
var
  i : integer;
const
  LDefaultExtension = '*.csv';
begin
  FindAllFiles(AStrings, DesignFolder, LDefaultExtension, False);
  if AStrings.Count > 0 then begin
    for i := 0 to AStrings.Count -1 do begin
      AStrings[i] :=
        ExtractFileNameWithoutExt(ExtractFileNameOnly(AStrings[i]));
    end;
  end;
end;

procedure GetFontFilesFor(AStrings: TStrings);
var
  i : integer;
  LDefaultFolder : string;
const
  LExtension = '*.ttf';
  LFolder    = 'media';
  LSubFolder = 'fonts';
begin
  LDefaultFolder := ConcatPaths([LFolder, LSubFolder]);
  FindAllFiles(AStrings, LDefaultFolder, LExtension, False);
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
    Files.Clear;
    Files.Free;
  end;
end;

procedure FreeConfigurationFile;
begin
  if Assigned(ConfigurationFile) then begin
    ConfigurationFile.UpdateFile;
    CopyFile(Pool.ConfigurationFilename, Pool.BaseFilename+'.ini');
    ConfigurationFile.Free;
    ConfigurationFile := nil;
    Pool.ConfigurationFilename := '';
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
    LStringList.Clear;
    LStringList.Free;
  end;
end;

function NewConfigurationFile : string;
begin
  Result := Pool.BasePath + 'last_session.ini';
  if FileExists(Result) then
    DeleteFile(Result);

  FreeConfigurationFile;
  ConfigurationFile := TConfigurationFile.Create(Result);
  ConfigurationFile.CacheUpdates := True;
  ConfigurationFile.WriteInt64(_Main, 'RandSeed', RandSeed);
  ConfigurationFile.Invalidate;
end;

function LoadConfigurationFile(AFilename : string) : string;
begin
  if FileExists(AFilename) then begin
    FreeConfigurationFile;
    ConfigurationFile := TConfigurationFile.Create(AFilename);
    ConfigurationFile.CacheUpdates := True;
    RandSeed:= ConfigurationFile.ReadInt64(_Main, 'RandSeed', RandSeed);
    Result := AFilename;
  end else begin
    raise EFileNotFoundException.Create(AFilename);
  end;
end;


end.

