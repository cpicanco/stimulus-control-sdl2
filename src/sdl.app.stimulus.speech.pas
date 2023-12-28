{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimulus.speech;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , SDL2
  , sdl.app.graphics.rectangule
  , sdl.app.stimulus
  //, sdl.app.stimulus.typeable
  , sdl.app.graphics.toggle
  , sdl.app.events.abstract
  , sdl.app.audio
  , sdl.app.audio.recorder.devices
  ;

type

  { TSpeechStimulus }

  TSpeechStimulus = class(TStimulus)
  private
    //FTextFromVocalResponse : string;
    FRect : TSDL_Rect;
    FRecorder : TAudioRecorderComponent;
    FPlayback : TAudioPlaybackComponent;
    FRecorderButton : TToggleButton;
    FPlaybackButton : TToggleButton;
  protected
    function GetRect: TRectangule; override;
    function GetStimulusName : string; override;
    //procedure RecorderTerminated(Sender: TObject);
    //procedure PlaybackTerminated(Sender: TObject);
    procedure MouseUp(Sender: TObject; Shift: TCustomShiftState;
      X, Y: Integer); override;
    //procedure KeyUp;q
  public
    constructor Create; override;
    destructor Destroy; override;
    function IsCorrectResponse : Boolean; override;
    procedure DoResponse(AHuman: Boolean); override;
    procedure Load(AParameters : TStringList;
        AParent : TObject; ARect: TSDL_Rect); override;
    procedure Start; override;
    procedure Stop; override;
  end;

implementation

uses Controls
   , session.pool
   , sdl.app.output
   , sdl.app.controls.custom
   , session.constants.mts
   , session.strutils
   , session.strutils.mts
   , session.loggers.writerow.timestamp
   , session.parameters.global
   , forms.modal.speechvalidation
   ;

{ TSpeechStimulus }

procedure TSpeechStimulus.DoResponse(AHuman: Boolean);
var
  LName: String;
begin
  inherited DoResponse(AHuman);
  LName := GetID.ToString.Replace(#9, '-').Replace(#32, '-');
  FRecorder.SaveToFile(Pool.DataResponsesBasePath+LName);
  FormManualSpeechValidation.ExpectedText := FCustomName;
end;

function TSpeechStimulus.GetRect: TRectangule;
begin
  Result := FRecorderButton as TRectangule;
end;

function TSpeechStimulus.GetStimulusName: string;
begin
  Result := 'Speech' + #9 + FCustomName;
end;

procedure TSpeechStimulus.MouseUp(Sender: TObject; Shift: TCustomShiftState; X,
  Y: Integer);
begin
  if Sender = FRecorderButton then begin
    if FRecorder.CanRecord then begin
      FRecorder.StartRecording(FRecorderButton);
      FRecorderButton.Toggle;
    end;
    Exit;
  end;

  if Sender = FPlaybackButton then begin
    if FRecorder.HasRecording then begin
      FPlayback.StartPlayback(FPlaybackButton);
      FPlaybackButton.Toggle;
    end;
  end;
end;

constructor TSpeechStimulus.Create;
begin
  inherited Create;
  FPlaybackButton := TToggleButton.Create;
  FPlaybackButton.Owner := Self as TObject;
  FRecorderButton := TToggleButton.Create;
  FRecorderButton.Owner := Self as TObject;

  Selectables.Add(FRecorderButton.AsISelectable);
  Selectables.Add(FPlaybackButton.AsISelectable);
end;

destructor TSpeechStimulus.Destroy;
begin
  FPlaybackButton.Free;
  FRecorderButton.Free;
  inherited Destroy;
end;

function TSpeechStimulus.IsCorrectResponse: Boolean;
begin
  if GlobalTrialParameters.ShowModalFormForSpeechResponses then begin
    Result := FormManualSpeechValidation.ShowModal = mrYes;
  end else begin
    Result := True;
  end;
  if not Result then begin
    Timestamp(
      'Incorrect.Response' + #9 + FormManualSpeechValidation.EditSpeech.Text);
  end;
end;

procedure TSpeechStimulus.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
const
  LRecordButtonOn  : string = 'RecordButtonOn';
  LRecordButtonOff : string = 'RecordButtonOff';
  LPlayButtonOn  : string = 'PlayButtonOn';
  LPlayButtonOff : string = 'PlayButtonOff';
begin
  //inherited Load(AParameters, AParent, ARect);
  FRect := ARect;
  FCustomName := GetWordValue(AParameters, IsSample, Index);

  SDLAudio.RecorderDevice.Clear;
  FRecorder := SDLAudio.RecorderDevice.Recorder;
  FPlayback := SDLAudio.RecorderDevice.Playback;

  if FPlayback.Opened then begin
    FPlaybackButton.LoadFromFile(
      AsAsset(LPlayButtonOff), AsAsset(LPlayButtonOn));
    FPlaybackButton.BoundsRect := ARect;
    FPlaybackButton.Parent := TSDLControl(AParent);
    //FPlaybackButton.OnMouseDown := @MouseDown;
    FPlaybackButton.OnMouseUp := @MouseUp;
    SDLAudio.RecorderDevice.Append(FPlaybackButton);
  end;

  if FRecorder.Opened then begin
    FRecorderButton.LoadFromFile(
      AsAsset(LRecordButtonOff), AsAsset(LRecordButtonOn));
    FRecorderButton.BoundsRect := ARect;
    FRecorderButton.Sibling := FPlaybackButton;
    FRecorderButton.Parent := TSDLControl(AParent);
    //FRecordButton.OnMouseDown := @MouseDown;
    FRecorderButton.OnMouseUp := @MouseUp;
    SDLAudio.RecorderDevice.Append(FRecorderButton);
  end;
end;

procedure TSpeechStimulus.Start;
begin
  if IsSample then begin
    { do nothing }
  end else begin
    if Assigned(FRecorderButton) then begin
      FRecorderButton.Show;
      FRecorderButton.Enabled := True;
    end;
    if Assigned(FPlaybackButton) then begin
      FPlaybackButton.Show;
      FPlaybackButton.Enabled := False;
    end;
  end;
end;

procedure TSpeechStimulus.Stop;
begin
  if IsSample then begin
    { do nothing }
  end else begin
    if Assigned(FRecorderButton) then begin
      FRecorderButton.Hide;
    end;
    if Assigned(FPlaybackButton) then begin
      FPlaybackButton.Hide;
    end;
  end;
end;

end.

