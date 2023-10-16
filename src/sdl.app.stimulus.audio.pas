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
  //, fgl
  , sdl.app.graphics.rectangule
  , sdl.app.audio.contract
  , sdl.app.stimulus
  , sdl.app.graphics.picture
  , sdl.app.events.abstract
  ;

type

  { TAudioStimulus }

  TAudioStimulus = class(TStimulus)
  private
    FSound : ISound;
    FPicture : TPicture;
  protected
    function GetStimulusName : string; override;
    function GetRect: TRectangule; override;
    procedure SetRect(AValue: TRectangule); override;
    procedure SoundFinished(Sender: TObject);
    procedure MouseDown(Sender: TObject; Shift: TCustomShiftState;
      X, Y: Integer); override;
    procedure MouseEnter(Sender: TObject); override;
    procedure MouseExit(Sender: TObject); override;
  public
    destructor Destroy; override;
    procedure Load(AParameters : TStringList;
        AParent : TObject; ARect: TSDL_Rect); override;
    procedure Start; override;
    procedure Stop; override;
  end;

implementation

uses sdl.app.audio
   , sdl.app.renderer.custom
   , session.loggers.writerow.timestamp
   , session.pool
   , session.constants.mts
   , session.strutils
   , session.strutils.mts;

{ TAudioStimulus }

function TAudioStimulus.GetStimulusName: string;
begin
  if IsSample then begin
    Result := 'Audio.Sample' + #9 + FWord;
  end else begin
    Result := 'Audio.Comparison' + #9 + FWord;
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
  if IsSample then begin
    Timestamp('Stop.' + GetStimulusName);
    DoResponse;
    FSound.SetOnStop(nil);
  end else begin
    DoResponse;
  end;
end;

procedure TAudioStimulus.MouseDown(Sender: TObject; Shift: TCustomShiftState;
  X, Y: Integer);
begin
  if SDLAudio.Playing then begin
    { do nothing }
  end else begin
    if Assigned(OnMouseDown) then
      OnMouseDown(Self, Shift, X, Y);

    Timestamp('Start.' + GetStimulusName);
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

destructor TAudioStimulus.Destroy;
begin
  //SDLAudio.UnregisterChannel(FSound);
  //FSound.Free;
  inherited Destroy;
end;

procedure TAudioStimulus.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
const
  LAudioPicture : string = 'AudioPicture'+IMG_EXT;
begin
  FPicture := TPicture.Create(Self);
  FPicture.LoadFromFile(Assets(LAudioPicture));
  FPicture.BoundsRect := ARect;
  FPicture.Parent := TCustomRenderer(AParent);
  FPicture.OnMouseDown := @MouseDown;
  FPicture.OnMouseEnter := @MouseEnter;
  FPicture.OnMouseExit := @MouseExit;

  FWord := GetWordValue(AParameters, IsSample, Index);
  FSound := SDLAudio.LoadFromFile(FWord+'.wav');
  FSound.SetOnStop(@SoundFinished);
end;

procedure TAudioStimulus.Start;
begin
  if IsSample then begin
    Timestamp('Start.' + GetStimulusName);
    FSound.Play;
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

