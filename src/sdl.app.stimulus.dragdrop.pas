{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus.dragdrop;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , sdl.app.graphics.picture.dragdrop
  , sdl.app.stimulus
  , sdl.app.events.abstract
  ;

type

  { TDragDropableStimulus }

  TDragDropableStimulus = class(TStimulus)
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
  end;

implementation

uses
   sdl.app.controls.custom
   , session.strutils.mts
   , session.strutils;

{ TDragDropableStimulus }

function TDragDropableStimulus.GetStimulusName: string;
begin
  if IsSample then begin
    Result := 'Picture.Sample' + #9 + FCustomName;
  end else begin
    Result := 'Picture.Comparison' + #9 + FCustomName;
  end;
end;

procedure TDragDropableStimulus.MouseUp(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  DoResponse(True);
end;

constructor TDragDropableStimulus.Create;
begin
  inherited Create;
  FPicture := TPicture.Create;

  Selectables.Add(FPicture.AsISelectable);
end;

destructor TDragDropableStimulus.Destroy;
begin
  FPicture.Free;
  inherited Destroy;
end;

procedure TDragDropableStimulus.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
begin
  FCustomName := GetWordValue(AParameters, IsSample, Index);
  FPicture.LoadFromFile(AsImage(FCustomName));
  FPicture.CustomName := FCustomName;
  FPicture.BoundsRect := ARect;
  FPicture.Parent := TSDLControl(AParent);
  FPicture.OnMouseUp := @MouseUp;
end;

procedure TDragDropableStimulus.Start;
begin
  FPicture.Show;
end;

procedure TDragDropableStimulus.Stop;
begin
  FPicture.Hide;
end;

end.

