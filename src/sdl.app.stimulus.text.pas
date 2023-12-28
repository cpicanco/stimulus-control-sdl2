{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus.text;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , sdl.app.audio.contract
  , sdl.app.graphics.text
  , sdl.app.stimulus
  , sdl.app.events.abstract
  , session.strutils.mts
  ;

type

  //TTextList = specialize TList<TText>;

  { TTextStimulus }

  TTextStimulus = class(TStimulus)
    private
      FPrompt : ISound;
      FText : TText;
      FHasPrompt : Boolean;
    protected
      function GetStimulusName : string; override;
      procedure MouseDown(Sender: TObject; Shift: TCustomShiftState;
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
  sdl.app.audio
  , session.parameters.global
  , sdl.app.controls.custom
  , session.constants.mts;

{ TTextStimuli }

function TTextStimulus.GetStimulusName: string;
begin
  if IsSample then begin
    Result := 'Text.Sample' + #9 + FCustomName;
  end else begin
    Result := 'Text.Comparison' + #9 + FCustomName;
  end;
end;

procedure TTextStimulus.MouseDown(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  DoResponse(True);
end;

constructor TTextStimulus.Create;
begin
  inherited Create;
  FText := TText.Create;
  FPrompt := nil;
  FHasPrompt := False;

  Selectables.Add(FText.AsISelectable);
end;

destructor TTextStimulus.Destroy;
begin
  FPrompt := nil;
  FText.Free;
  inherited Destroy;
end;

procedure TTextStimulus.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
begin
  FCustomName := GetWordValue(AParameters, IsSample, Index);
  FText.FontName := GlobalTrialParameters.FontName;
  //FText.FontSize := 50;
  FText.Load(FCustomName);
  FText.CentralizeWith(ARect);
  FText.Parent := TSDLControl(AParent);
  FText.OnMouseDown := @MouseDown;

  FHasPrompt := HasPrompt(AParameters);
  if FHasPrompt then begin
    if IsSample then
      FPrompt := SDLAudio.SoundFromName(
        GlobalTrialParameters.AudioPromptForText);
  end;
end;

procedure TTextStimulus.Start;
begin
  if FHasPrompt then begin
    if IsSample then
      FPrompt.Play;
  end;
  FText.Show;
end;

procedure TTextStimulus.Stop;
begin
  FText.Hide;
end;

end.

