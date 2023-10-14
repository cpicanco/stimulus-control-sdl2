{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit picanco.experiments.images;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, FileUtil;


implementation

uses session.pool, sdl.app.graphics.picture, picanco.experiments.constants,
  picanco.experiments.words.types;

function GetAllImages: TStringList;
var
  LMediaPath : string = 'base';
begin
  LMediaPath := ConcatPaths([Pool.RootMedia, LMediaPath]);
  Result := FindAllFiles(LMediaPath, '*'+IMG_EXT, False);
end;

procedure CopyImagesToMediaPath(AFileNames: array of string);
var
  i: integer;
  SrcFile, DstFile: string;
  AllImages : TStringList;
begin
  AllImages := GetAllImages;
  if AllImages.Count < Length(AFileNames) then begin
    AllImages.Free;
    Exit;
  end;

  try
    for i := 0 to AllImages.Count-1 do
      AllImages.Exchange(i, Random(AllImages.Count));

    for i := Low(AFileNames) to High(AFileNames) do begin
      SrcFile := AllImages[i];
      DstFile := ConcatPaths([Pool.RootMedia, AFileNames[i]]);
      if FileExists(SrcFile) and (not FileExists(DstFile)) then
        CopyFile(SrcFile, DstFile);
    end;
  finally
    AllImages.Free;
  end;
end;

procedure E1NewImages;
var
  LImages : array of string = nil;
  n: Integer;
  Code : TAlphaNumericCode;
begin
  n := 0;
  SetLength(LImages, n);
  for Code in E1WordsWithImagesRange do begin
      SetLength(LImages, Length(LImages)+1);
      LImages[n] := E1WordsWithImages[Code] + IMG_EXT;
      Inc(n);
  end;
  CopyImagesToMediaPath(LImages);
end;

initialization
  E1NewImages;

end.

