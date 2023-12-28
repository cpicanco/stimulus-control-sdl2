{
  Stimulus Control
  Copyright (C) 2014-2023 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimuli.calibration.pupil;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , sdl.app.graphics.text
  , sdl.app.stimuli
  , sdl.app.stimuli.contract
  , pupil.client;

type

  { TPupilCalibrationStimuli }

  TPupilCalibrationStimuli = class sealed (TStimuli)
    private
      FText : TText;
      procedure HideCalibrationFailedMessage(Sender: TObject);
      procedure PupilCalibrationSuccessful(Sender: TObject;
        APupilMessage : TPupilMessage);
      procedure PupilCalibrationFailed(Sender: TObject;
        APupilMessage : TPupilMessage);
      procedure StartPupilCalibration;
    public
      constructor Create; override;
      destructor Destroy; override;
      function AsInterface : IStimuli;
      procedure DoExpectedResponse; override;
      procedure Load(AParameters : TStringList;
        AParent : TObject); override;
      procedure Start; override;
      procedure Stop; override;
  end;

implementation

uses sdl.app.video.methods, sdl.app.controls.custom;

{ TPupilCalibrationStimuli }

procedure TPupilCalibrationStimuli.HideCalibrationFailedMessage(
  Sender: TObject);
begin
  FText.Hide;
end;

procedure TPupilCalibrationStimuli.PupilCalibrationSuccessful(Sender: TObject;
  APupilMessage: TPupilMessage);
begin
  RaiseWindow;
  FText.Load('A calibragem ocorreu.');
  FText.Centralize;
end;

procedure TPupilCalibrationStimuli.PupilCalibrationFailed(Sender: TObject;
  APupilMessage: TPupilMessage);
begin
  RaiseWindow;
  FText.Load('A calibragem falhou.');
  FText.Centralize;
end;

procedure TPupilCalibrationStimuli.StartPupilCalibration;
begin
  PupilClient.Request(REQ_SHOULD_START_CALIBRATION, True);
end;

constructor TPupilCalibrationStimuli.Create;
begin
  inherited Create;
  FText := TText.Create;
end;

destructor TPupilCalibrationStimuli.Destroy;
begin
  FText.Free;
  inherited Destroy;
end;

function TPupilCalibrationStimuli.AsInterface: IStimuli;
begin
  Result := Self as IStimuli;
end;

procedure TPupilCalibrationStimuli.DoExpectedResponse;
begin

end;

procedure TPupilCalibrationStimuli.Load(AParameters: TStringList; AParent: TObject);
begin
  inherited Load(AParameters, AParent);
  FText.FontName := 'Raleway-Regular';
  FText.FontSize := 50;
  FText.Parent := TSDLControl(AParent);

  PupilClient.OnCalibrationSuccessful := @PupilCalibrationSuccessful;
  PupilClient.OnCalibrationFailed := @PupilCalibrationFailed;
end;

procedure TPupilCalibrationStimuli.Start;
begin
  StartPupilCalibration;
end;

procedure TPupilCalibrationStimuli.Stop;
begin

end;

end.

