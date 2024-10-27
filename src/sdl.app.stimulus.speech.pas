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
  , sdl.app.audio
  , sdl.app.audio.recorder.devices
  ;

type

  { TSpeechStimulus }

  TSpeechStimulus = class(TStimulus)
  private
    FHasConsequence : Boolean;
    FRectangule : TRectangule;
    FRecorder : TAudioRecorderComponent;
    procedure RecordingFinished(Sender: TObject);
    procedure RecordingStopped(Sender: TObject);
  protected
    function GetRect: TRectangule; override;
    function GetStimulusName : string; override;
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
   //, sdl.app.output
   , session.constants.trials
   , session.strutils.mts
   , session.loggers.writerow.timestamp
   , session.loggers.types
   , session.parameters.global
   , forms.modal.speechvalidation
   , forms.speechvalidation
   ;

{ TSpeechStimulus }

{
  Without speech-to-text capabilities,
  it is necessary to manually record
  speech responses. Some computer-assisted
  annotation support has been implemented:

  1) Empty ExpectedText entries represent
  incorrect responses (or non-responses,
  as one might consider them) triggered
  by a limited hold. These are automatically
  recorded.

  2) Expected correct responses are
  auto-completed. Therefore, if a
  participant emits a correct response,
  the experimenter can promptly
  press Enter to record it.

  3) Common incorrect responses (e.g.,
  "I don't know") are triggered by
  a reserved keyboard key (- key).

  Other incorrect responses may be typed
  using the keyboard.

  Thus, manual annotation may consistently
  take varying amounts of time in some cases
  (for example, 2 and 4), with a slightly
  longer time for wrong responses. Therefore,
  for assessment trials programmed to have
  no differential consequences, computer-assisted
  annotation must be non-blocking to avoid
  consistent differential timing.

  Non-blocking annotation during a trial
  requires asynchronous writing in the
  .timestamps file. This may result in
  unordered lines regarding monotonic time,
  trials, and blocks. One may sort them
  by time post-facto to recover the correct
  event ordering.

  Non-blocking annotation between trials
  in the .data file is not covered.
}
procedure TSpeechStimulus.DoResponse(AHuman: Boolean);
var
  LName: String;
  LEvent: TTimestampedEvent;
begin
  LEvent := TimestampedEvent;
  DoResponseIncrement;

  if AHuman then begin
    LEvent.Code := 'Stimulus.Response.Speech';
  end else begin
    LEvent.Code := 'Stimulus.Robot.Response.Speech';
  end;
  LEvent.Annotation := FCustomName;

  Timestamp(LEvent);

  LName := 'Speech-'+GetID.ToSpeechString;
  FRecorder.SaveToFile(Pool.DataResponsesBasePath+LName);

  if AHuman then begin
    if FHasConsequence then begin
      FormManualSpeechValidation.ExpectedText := FCustomName;
    end else begin
      LEvent.Annotation := FCustomName;
      FormSpeechValidationQueue.ExpectedText := LEvent;
    end;
  end else begin
    { do nothing }
  end;

  if Assigned(OnResponse) then
    OnResponse(Self);
end;

procedure TSpeechStimulus.RecordingFinished(Sender: TObject);
begin
  DoResponse(False);
end;

procedure TSpeechStimulus.RecordingStopped(Sender: TObject);
begin
  DoResponse(True);
end;

function TSpeechStimulus.GetRect: TRectangule;
begin
  Result := FRectangule;
end;

function TSpeechStimulus.GetStimulusName: string;
begin
  if IsSample then begin // currently not using speech as samples...
    Result := 'Speech.Sample' + #9 + FCustomName;
  end else begin
    Result := 'Speech.Comparison' + #9 + FCustomName;
  end;
end;

constructor TSpeechStimulus.Create;
begin
  inherited Create;
  FormManualSpeechValidation.ExpectedText := '';

  FRectangule := TRectangule.Create;
end;

destructor TSpeechStimulus.Destroy;
begin
  FRectangule.Free;
  inherited Destroy;
end;

function TSpeechStimulus.IsCorrectResponse : Boolean;
var
  LExpectedResponse : string = '';
  LAnnotation : string = '';
begin
  if FHasConsequence then begin
    LExpectedResponse := FormManualSpeechValidation.ExpectedText;
  end else begin
    LExpectedResponse := FormSpeechValidationQueue.ExpectedText.Annotation;
  end;

  if LExpectedResponse.IsEmpty then begin
    Result := False;
  end else begin
    if GlobalTrialParameters.ShowModalFormForSpeechResponses then begin
      if FHasConsequence then begin
        with FormManualSpeechValidation do begin
          ShowModal; // blocking
          Result := EditSpeech.Text = ExpectedText; // change in real-time
        end;
      end else begin
        Result := True; // unchanged in real-time, requires post-fact analysis
      end;
    end else begin
      Result := True; // unchanged in real-time, requires post-fact analysis
    end;
  end;

  if not Result then begin
    if FHasConsequence then begin
      LAnnotation :=
        LExpectedResponse + '-' + FormManualSpeechValidation.EditSpeech.Text;
    end else begin
      { do nothing }
    end;
    Timestamp('Incorrect.Response', LAnnotation);
  end;
end;

procedure TSpeechStimulus.Load(AParameters: TStringList; AParent: TObject;
  ARect: TSDL_Rect);
begin
  FRectangule.BoundsRect := ARect;
  // FRectangule.Visible := False;
  // FRectangule.Parent := AParent; // speech do not have a visible square
  FCustomName := GetWordValue(AParameters, IsSample, Index);
  FHasConsequence := AParameters.Values[TrialKeys.HasConsequenceKey].ToBoolean;

  SDLAUdio.RecorderDevice.Recorder.Clear;
  SDLAUdio.RecorderDevice.OnStopped := @RecordingStopped;
  SDLAUdio.RecorderDevice.OnFinished := @RecordingFinished;
  FRecorder := SDLAudio.RecorderDevice.Recorder;
end;

procedure TSpeechStimulus.Start;
begin
  if IsSample then begin
    { do nothing }
  end else begin
    if FRecorder.CanRecord then begin
      FRecorder.StartRecording(Self);
    end;
  end;
end;

procedure TSpeechStimulus.Stop;
begin
  { do nothing }
end;

end.

