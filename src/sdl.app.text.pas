{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.text;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Generics.Collections
  , ctypes
  , sdl2_ttf
  ;

type
    TFontData = record
      Name : string;
      Size : cint;
      Font : PTTF_Font;
    end;

    TFontCollection = specialize TDictionary<string, TFontData>;

    { TSDLText }

    TSDLText = class
    private
      FFontCollention : TFontCollection;
    protected
      procedure LoadFromFile(AFilename : string);
    public
      constructor Create; reintroduce;
      destructor Destroy; override;
      procedure SetupFonts;
      function Get(AFontName: string) : TFontData;
    end;

var
  SDLText : TSDLText;

implementation

uses
  FileUtil, LazFileUtils
  , sdl.app.video.methods
  //, sdl.app.output
  , session.pool
  , session.parameters.global;

{ TSDLText }

constructor TSDLText.Create;
begin
  FFontCollention := TFontCollection.Create;
  if TTF_Init < -1 then begin
    raise Exception.Create('SDL2_TTF not initialized:'+ TTF_GetError^);
  end else begin
    //Print('SDL2_TTF initialized.');
  end;
end;

destructor TSDLText.Destroy;
var
  i : integer;
begin
  for i := 0 to FFontCollention.Count -1 do begin
    TTF_CloseFont(FFontCollention.Values.ToArray[i].Font);
  end;
  FFontCollention.Free;
  inherited Destroy;
end;

procedure TSDLText.SetupFonts;
var
  TTFFiles: TStringList;
  LFilename: string;
begin
  TTFFiles := TStringList.Create;
  try
    FindAllFiles(TTFFiles, Pool.FontsRootBasePath, '*.ttf', True);
    for LFilename in TTFFiles do
      LoadFromFile(LFilename);
  finally
    TTFFiles.Clear;
    TTFFiles.Free;
  end;
end;

function TSDLText.Get(AFontName: string): TFontData;
begin
  Result := FFontCollention[AFontName];
end;

function GetFontSize: cint32;
const
  ReferenceWidth = 1280;
  ReferenceFontSize = 150;
  MinimumFontSize = 90;
begin
  Result := Round(MonitorFromWindow.w / ReferenceWidth * ReferenceFontSize);
  if MonitorFromWindow.h < MonitorFromWindow.w then
    Result := Round(Result * (MonitorFromWindow.h / MonitorFromWindow.w));

  if Result < MinimumFontSize then
    Result := MinimumFontSize;
end;

procedure TSDLText.LoadFromFile(AFilename: string);
var
  LFilename : PAnsiChar;
  LPTTFFont : PTTF_Font;
  LFontData : TFontData = (Name : ''; Size : 0; Font : nil);
begin
  LFilename := PAnsiChar(AFilename);
  if GlobalTrialParameters.FontSize = 0 then begin
    LFontData.Size := GetFontSize;
  end else begin
    LFontData.Size := GlobalTrialParameters.FontSize;
  end;
  LPTTFFont := TTF_OpenFont(LFilename, LFontData.Size);
  if LPTTFFont <> nil then begin
    LFontData.Name := ExtractFileNameWithoutExt(ExtractFileNameOnly(AFilename));
    LFontData.Font := LPTTFFont;
    FFontCollention.Add(LFontData.Name, LFontData);
  end;
end;

end.

