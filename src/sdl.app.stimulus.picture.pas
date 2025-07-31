{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

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
      procedure MouseUp(Sender: TObject; Shift: TCustomShiftState;
        X, Y: Integer); override;
    public
      constructor Create; override;
      destructor Destroy; override;
      procedure Load(AParameters : TStringList;
        AParent : TObject; ARect: TSDL_Rect); override;
      procedure Start; override;
      procedure Stop; override;
      procedure Hide; override;
  end;

implementation

uses
   sdl.app.controls.custom
   , session.strutils.mts
   , session.strutils;

{ TPictureStimulus }

function TPictureStimulus.GetStimulusName: string;
begin
  if IsSample then begin
    Result := 'Picture.Sample' + #9 + FCustomName;
  end else begin
    Result := 'Picture.Comparison' + #9 + FCustomName;
  end;
end;

procedure TPictureStimulus.MouseUp(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  DoResponse(True);
end;

constructor TPictureStimulus.Create;
begin
  inherited Create;
  FPicture := TPicture.Create;

  Selectables.Add(FPicture.AsISelectable);
end;

destructor TPictureStimulus.Destroy;
begin
  FPicture.Free;
  inherited Destroy;
end;

procedure TPictureStimulus.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
begin
  FCustomName := GetWordValue(AParameters, IsSample, Index);
  FPicture.LoadFromFile(AsImage(FCustomName));
  FPicture.CustomName := FCustomName;
  FPicture.BoundsRect := ARect;
  FPicture.Parent := TSDLControl(AParent);
  FPicture.OnMouseUp := @MouseUp;
end;

procedure TPictureStimulus.Start;
begin
  FPicture.Show;
end;

procedure TPictureStimulus.Stop;
begin
  FPicture.Hide;
end;

procedure TPictureStimulus.Hide;
begin
  inherited Hide;
  FPicture.Hide;
end;

end.

