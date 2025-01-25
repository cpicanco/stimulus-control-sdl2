{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

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
  , sdl.app.graphics.animatedtext
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
      FText : TAnimatedText;
      FHasPrompt : Boolean;
      procedure PromptStopped(Sender: TObject);
      procedure PromptStarted(Sender: TObject);
      function GetStimulusNamePrefix : string;
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
  //sdl.app.output,
  sdl.app.controls.custom,
  session.parameters.global,
  session.loggers.writerow.timestamp;

{ TTextStimuli }

procedure TTextStimulus.PromptStopped(Sender: TObject);
var
  LCustomName : string;
  LSound : ISound;
begin
  LSound := Sender as ISound;
  LCustomName := LSound.GetCustomData;

  LSound.SetOnStop(nil);

  Timestamp(GetStimulusNamePrefix+'.Prompt.Stop', LCustomName);

  // starts CD recording
  if Assigned(Sibling) then begin
    FText.Animate;
    Sibling.Start;
  end;
end;

procedure TTextStimulus.PromptStarted(Sender: TObject);
var
  LSound : ISound;
  LCustomName : string;
begin
  LSound := Sender as ISound;
  LCustomName := LSound.GetCustomData;
  Timestamp(GetStimulusNamePrefix+'.Prompt.Start', LCustomName);
end;

function TTextStimulus.GetStimulusNamePrefix: string;
begin
  if IsSample then begin
    Result := 'Text.Sample';
  end else begin
    Result := 'Text.Comparison';
  end;
end;

function TTextStimulus.GetStimulusName: string;
begin
  Result := GetStimulusNamePrefix + #9 + FCustomName
end;

procedure TTextStimulus.MouseUp(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  if FHasPrompt then begin
    if IsSample then begin
      if FPrompt.Playing then begin
        Exit;
      end;
    end;
  end;
  DoResponse(True);
end;

constructor TTextStimulus.Create;
begin
  inherited Create;
  FText := TAnimatedText.Create;
  //FPrompt := nil;
  FHasPrompt := False;
end;

destructor TTextStimulus.Destroy;
begin
  //FPrompt := nil;
  FText.Free;
  inherited Destroy;
end;

procedure TTextStimulus.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
begin
  FCustomName := GetWordValue(AParameters, IsSample, Index);

  FText.FontName := GetFontName(AParameters);
  FText.Load(FCustomName);
  FText.CentralizeWith(ARect);
  FText.CustomName := FCustomName;
  Selectables.Add(FText.AsISelectable);
  FText.Parent := TSDLControl(AParent);
  FText.Fit(ARect);
  FText.OnMouseUp := @MouseUp;

  FHasPrompt := HasTextPrompt(AParameters);
  if FHasPrompt and IsSample then begin
    FPrompt := GetAudioPromptForText(AParameters);
    FPrompt.SetOnStop(@PromptStopped);
    FPrompt.SetOnStart(@PromptStarted);
    FPrompt.SetCustomData(FCustomName);
  end;
end;

procedure TTextStimulus.Start;
begin
  FText.Show;
  if FHasPrompt then begin
    if IsSample then begin
      FPrompt.Play;
    end;
  end;
end;

procedure TTextStimulus.Stop;
begin
  FText.Hide;
end;

procedure TTextStimulus.Hide;
begin
  inherited Hide;
  FText.Hide;
end;

end.

