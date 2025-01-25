{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit picanco.experiments.images;

{$mode ObjFPC}{$H+}

interface

procedure E1CopyRandomImagesToParticipantFolder;

implementation

uses
  Classes,
  SysUtils,
  FileUtil,
  session.pool,
  session.strutils,
  sdl.app.graphics.picture,
  picanco.experiments.constants,
  picanco.experiments.words.types;

function GetSrcImages: TStringList;
var
  LMediaPath : string = 'base';
begin
  LMediaPath := ConcatPaths([Pool.ImageRootBasePath, LMediaPath]);
  Result := FindAllFiles(LMediaPath, '*'+IMG_EXT, False);
end;

function NewImagesFolder : string;
var
  LMediaPath : string = 'pool';
begin
  Result := ConcatPaths([Pool.ImageRootBasePath, LMediaPath]);
end;

function HasDstImages: Boolean;
var
  LDstImages : TStringList;
begin
  LDstImages := FindAllFiles(MeaningfulImageFolder, '*'+IMG_EXT, False);
  try
    Result := LDstImages.Count = 0;
  finally
    LDstImages.Free;
  end;
end;

procedure CopyImagesToMediaPath(ADstImages: TStringList);
var
  i: integer;
  Image, SrcFile, DstFile: string;
  LSrcImages : TStringList;
begin
  LSrcImages := GetSrcImages;
  try
    if LSrcImages.Count < ADstImages.Count then begin
      raise ERangeError.Create('CopyImagesToMediaPath');
    end;

    for i := 0 to LSrcImages.Count-1 do
      LSrcImages.Exchange(i, Random(LSrcImages.Count));

    for i := 0 to ADstImages.Count-1 do begin
      SrcFile := LSrcImages[i];
      DstFile := ADstImages[i];
      if FileExists(SrcFile) and (not FileExists(DstFile)) then
        CopyFile(SrcFile, DstFile);
    end;

    CopyFile(MeaningfulImageFolder+'quadrado'+IMG_EXT, AsImage('X1')+IMG_EXT);
    CopyFile(MeaningfulImageFolder+'estrela'+IMG_EXT, AsImage('X2')+IMG_EXT);

    for Image in E1WordsWithNewImages do begin
      SrcFile := ConcatPaths([NewImagesFolder, Image+IMG_EXT]);
      DstFile := AsImage(Image)+IMG_EXT;
      CopyFile(SrcFile, DstFile);
    end;
  finally
    LSrcImages.Clear;
    LSrcImages.Free;
  end;
end;

procedure E1CopyRandomImagesToParticipantFolder;
var
  LDstImages : TStringList;
  Code : TAlphaNumericCode;
begin
  if not HasDstImages then begin
    if ForceDirectories(ImageFolder) then begin
      LDstImages := TStringList.Create;
      LDstImages.Sorted := False;
      try
        for Code in E1WordsWithImagesRange do begin
          LDstImages.Append(AsImage(E1WordsWithImages[Code])+IMG_EXT);
        end;
        CopyImagesToMediaPath(LDstImages);
      finally
        LDstImages.Free;
      end;
    end;
  end;
end;

end.

