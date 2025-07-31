{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.strutils;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils;

  function KeyValue(AKey, AValue: string;
    ALineEnding : string = LineEnding) : string;
  function AsAsset(ABasename: string) : string;
  function AsImage(ABasename: string) : string;
  function AsAudio(ABasename: string) : string;
  function AsMarker(ABasename: string) : string;
  function AsInstruction(ABasename: string) : string;

  function ImageFolder : string;
  function MeaningfulImageFolder : string;
  function DesignFolder : string;
  function TestModeFolder : string;

  function AsPath(A, B : string) : string; overload;
  function AsPath(A : string) : string; overload;
  function ArrayToStr(AChars: array of char; Len: SizeInt): string;


implementation

uses session.pool;

function KeyValue(AKey, AValue: string; ALineEnding: string): string;
begin
  Result := AKey + '=' + AValue + ALineEnding;
end;

function AsAsset(ABasename: string): string;
begin
  Result := Pool.AssetsRootBasePath+ABasename;
end;

function AsImage(ABasename: string): string;
begin
  Result := ConcatPaths([
    Pool.ImageRootBasePath, Pool.ImageBasePath]) + ABasename;
end;

function AsAudio(ABasename: string): string;
begin
  Result := ConcatPaths([
    Pool.AudioRootBasePath, Pool.AudioBasePath]) + ABasename;
end;

function AsMarker(ABasename: string): string;
begin
  Result := ConcatPaths([
    Pool.ImageRootBasePath, AsPath('markers')]) + ABasename;
end;

function AsInstruction(ABasename: string): string;
begin
  Result := ConcatPaths([
    Pool.MediaRootBasePath, AsPath('instructions')]) + ABasename;
end;

function AsPath(A: string): string;
begin
  Result := IncludeTrailingPathDelimiter(A);
end;

function ImageFolder: string;
begin
  Result := ConcatPaths([
    Pool.ImageRootBasePath, Pool.ImageBasePath]);
end;

function MeaningfulImageFolder : string;
begin
  Result := ConcatPaths([
    Pool.ImageRootBasePath, AsPath('meaningful')]);
end;

function DesignFolder: string;
begin
  Result := ConcatPaths([
    Pool.DesignRootBasePath, Pool.DesignBasePath]);
end;

function TestModeFolder: string;
begin
  Result := ConcatPaths([Pool.BaseDataPath, Pool.BaseFileName]);
end;

function AsPath(A, B: string): string;
begin
  Result := AsPath(ConcatPaths([A, B]));
end;

function ArrayToStr(AChars: array of char; Len: SizeInt): string;
begin
  SetString(Result, PChar(@AChars[0]), Len);
end;

end.

