{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus.audio;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , sdl.app.graphics.rectangule
  , sdl.app.audio.contract
  , sdl.app.audio.loops
  , sdl.app.stimulus
  , sdl.app.graphics.picture
  , sdl.app.graphics.text
  , sdl.app.events.abstract
  ;

type

  { TAudioStimulus }

  TAudioStimulus = class(TStimulus)
  private
    FLoops : TSoundLoop;
    FSound : ISound;
    FPicture : TPicture;
    FText    : TText;
  protected
    function GetStimulusName : string; override;
    function GetRect: TRectangule; override;
    procedure SetRect(AValue: TRectangule); override;
    procedure SoundFinished(Sender: TObject);
    procedure SoundStart(Sender: TObject);
    procedure MouseDown(Sender: TObject; Shift: TCustomShiftState;
      X, Y: Integer); override;
    procedure MouseEnter(Sender: TObject); override;
    procedure MouseExit(Sender: TObject); override;
  public
    constructor Create; override;
    destructor Destroy; override;
    procedure Load(AParameters : TStringList;
        AParent : TObject; ARect: TSDL_Rect); override;
    procedure Start; override;
    procedure Stop; override;
  end;

implementation

uses sdl.app.audio
   , sdl.app.renderer.custom
   , session.parameters.global
   , session.loggers.writerow.timestamp
   , session.pool
   , session.constants.mts
   , session.strutils
   , session.strutils.mts;

{ TAudioStimulus }

function TAudioStimulus.GetStimulusName: string;
begin
  if IsSample then begin
    Result := 'Audio.Sample' + #9 + FCustomName;
  end else begin
    Result := 'Audio.Comparison' + #9 + FCustomName;
  end;
end;

function TAudioStimulus.GetRect: TRectangule;
begin
  Result := FPicture as TRectangule;
end;

procedure TAudioStimulus.SetRect(AValue: TRectangule);
begin
  FPicture.BoundsRect := AValue.BoundsRect;
end;

procedure TAudioStimulus.SoundFinished(Sender: TObject);
begin
  Timestamp('Stop.' + GetStimulusName);
  if IsSample then begin
    if ResponseID = 0 then begin
      DoResponse(False);
      OnResponse := nil;
    end;
  end else begin
    DoResponse(False);
  end;
end;

procedure TAudioStimulus.SoundStart(Sender: TObject);
begin
  Timestamp('Start.' + GetStimulusName);
end;

procedure TAudioStimulus.MouseDown(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  if SDLAudio.Playing then begin
    { do nothing }
  end else begin
    if Assigned(OnMouseDown) then
      OnMouseDown(Self, Shift, X, Y);

    Timestamp('Stimulus.Response.' + GetStimulusName);
    FSound.Play;
  end;
end;

procedure TAudioStimulus.MouseEnter(Sender: TObject);
begin
  if IsSample then begin
    { do nothing }
  end else begin
    //FButtonPicture.Show;
  end;
end;

procedure TAudioStimulus.MouseExit(Sender: TObject);
begin
  if IsSample then begin
    { do nothing }
  end else begin
    //FButtonPicture.Hide;
  end;
end;

constructor TAudioStimulus.Create;
begin
  inherited Create;
  FPicture := TPicture.Create;
  FPicture.Owner := Self;
  FLoops := TSoundLoop.Create;
  FText := TText.Create;
end;

destructor TAudioStimulus.Destroy;
begin
  //SDLAudio.UnregisterChannel(FSound);
  //FSound.Free;
  FText.Free;
  FLoops.Free;
  FPicture.Free;
  inherited Destroy;
end;

procedure TAudioStimulus.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
const
  LAudioPicture : string = 'AudioPicture'+IMG_EXT;
begin
  FCustomName := GetWordValue(AParameters, IsSample, Index);
  if HasPrompt(AParameters) then begin
    FText.FontName := GlobalTrialParameters.FontName;
    //FText.FontSize := 50;
    FText.Load(FCustomName);
    FText.CentralizeWith(ARect);
    FText.Parent := TCustomRenderer(AParent);
    FText.OnMouseDown := @MouseDown;
  end else begin
    FPicture.LoadFromFile(Assets(LAudioPicture));
    FPicture.BoundsRect := ARect;
    FPicture.Parent := TCustomRenderer(AParent);
    FPicture.OnMouseDown := @MouseDown;
    FPicture.OnMouseEnter := @MouseEnter;
    FPicture.OnMouseExit := @MouseExit;
  end;

  FSound := SDLAudio.LoadFromFile(AudioFile(FCustomName));
  FSound.SetOnStop(@SoundFinished);
  FSound.SetOnStart(@SoundStart);
  FLoops.Sound := FSound;
end;

procedure TAudioStimulus.Start;
begin
  if IsSample then begin
    //FSound.Play;
    FPicture.Show;
  end else begin
    FPicture.Show;
  end;
end;

procedure TAudioStimulus.Stop;
begin
  FPicture.Hide;
end;

end.

