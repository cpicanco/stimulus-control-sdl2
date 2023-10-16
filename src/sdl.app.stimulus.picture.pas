{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus.picture;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , sdl.app.graphics.picture
  , sdl.app.stimulus
  , sdl.app.events.abstract
  ;

type

  { TPictureStimulus }

  TPictureStimulus = class(TStimulus)
    private
      FPicture : TPicture;
    protected
      function GetStimulusName : string; override;
      procedure MouseDown(Sender: TObject; Shift: TCustomShiftState;
        X, Y: Integer); override;
    public
      procedure Load(AParameters : TStringList;
        AParent : TObject; ARect: TSDL_Rect); override;
      procedure Start; override;
      procedure Stop; override;
  end;

implementation

uses
   sdl.app.renderer.custom
   , session.constants.mts
   , session.strutils.mts;

{ TPictureStimulus }

function TPictureStimulus.GetStimulusName: string;
begin
  if IsSample then begin
    Result := 'Picture.Sample' + #9 + FWord;
  end else begin
    Result := 'Picture.Comparison' + #9 + FWord;
  end;
end;

procedure TPictureStimulus.MouseDown(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  DoResponse;
end;

procedure TPictureStimulus.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
begin
  FPicture := TPicture.Create(Self);
  FWord := GetWordValue(AParameters, IsSample, Index);
  FPicture.LoadFromFile(FWord+IMG_EXT);
  FPicture.BoundsRect := ARect;
  FPicture.Parent := TCustomRenderer(AParent);
  FPicture.OnMouseDown := @MouseDown;
end;

procedure TPictureStimulus.Start;
begin
  FPicture.Show;
end;

procedure TPictureStimulus.Stop;
begin
  FPicture.Hide;
end;

end.

