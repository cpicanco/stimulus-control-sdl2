{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.stimuli.calibration;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils
  , sdl.app.graphics.text
  , sdl.app.stimuli
  , sdl.app.navigable.contract
  , sdl.app.navigator.contract
  , sdl.app.stimuli.contract
  , sdl.app.events.abstract;

type

  { TCalibrationStimuli }

  TCalibrationStimuli = class sealed (TStimuli, INavigable)
    private
      FNavigator : ITableNavigator;
      FText : TText;
      procedure SetNavigator(ANavigator: ITableNavigator);
      procedure UpdateNavigator;
      procedure CalibrationSuccessful(Sender: TObject);
      procedure CalibrationFailed(Sender: TObject);
      procedure MouseUp(Sender: TObject; Shift: TCustomShiftState;
        X, Y: Integer);
    public
      constructor Create; override;
      destructor Destroy; override;
      function AsInterface : IStimuli;
      function AsINavigable: INavigable; override;
      function IsStarter : Boolean; override;
      procedure DoExpectedResponse; override;
      procedure Load(AParameters : TStringList; AParent : TObject); override;
      procedure Start; override;
      procedure Stop; override;
  end;

implementation

uses
  sdl.app.video.methods,
  sdl.app.controls.custom,
  //sdl.app.selectable.contract,
  //sdl.app.selectable.list,
  session.loggers.writerow.timestamp,
  eye.tracker;

{ TCalibrationStimuli }

procedure TCalibrationStimuli.SetNavigator(ANavigator: ITableNavigator);
begin
  FNavigator := ANavigator;
end;

procedure TCalibrationStimuli.UpdateNavigator;
begin
  FSelectables.Add(FText.AsISelectable);
  FNavigator.UpdateNavigationControls(FSelectables);
end;

procedure TCalibrationStimuli.CalibrationSuccessful(Sender: TObject);
begin
  EyeTracker.StopCalibration;
  Timestamp(EyeTracker.TrackerClassName+'.StopCalibration');
  RaiseWindow;
  Finalize;
end;

procedure TCalibrationStimuli.CalibrationFailed(Sender: TObject);
begin
  EyeTracker.StopCalibration;
  Timestamp(EyeTracker.TrackerClassName+'.StopCalibration');
  RaiseWindow;

  if EyeTracker.IsFake then begin
    // force success when there is no real eye tracker
    EyeTracker.SetOnCalibrationFailed(@CalibrationSuccessful);
  end;

  FText.Load('A calibragem falhou. Tente novamente.');
  FText.Centralize;
  FText.Show;
end;

procedure TCalibrationStimuli.MouseUp(Sender: TObject;
  Shift: TCustomShiftState; X, Y: Integer);
begin
  FText.Hide;
  EyeTracker.StartCalibration;
end;

constructor TCalibrationStimuli.Create;
begin
  inherited Create;
  FText := TText.Create;
end;

destructor TCalibrationStimuli.Destroy;
begin
  EyeTracker.SetOnCalibrationSuccessful(nil);
  EyeTracker.SetOnCalibrationFailed(nil);
  FText.Free;
  inherited Destroy;
end;

function TCalibrationStimuli.AsInterface: IStimuli;
begin
  Result := Self as IStimuli;
end;

function TCalibrationStimuli.AsINavigable: INavigable;
begin
  Result := Self as INavigable;
end;

function TCalibrationStimuli.IsStarter: Boolean;
begin
  Result := True;
end;

procedure TCalibrationStimuli.DoExpectedResponse;
begin

end;

procedure TCalibrationStimuli.Load(AParameters: TStringList; AParent: TObject);
begin
  inherited Load(AParameters, AParent);
  FText.FontName := 'Raleway-Regular';
  FText.FontSize := 50;
  FText.Parent := TSDLControl(AParent);
  FText.OnMouseUp := @MouseUp;

  EyeTracker.SetOnCalibrationSuccessful(@CalibrationSuccessful);
  EyeTracker.SetOnCalibrationFailed(@CalibrationFailed);
end;

procedure TCalibrationStimuli.Start;
begin
  EyeTracker.StartCalibration;
  Timestamp(EyeTracker.TrackerClassName+'.StartCalibration');
end;

procedure TCalibrationStimuli.Stop;
begin
  FText.Hide;
end;

end.

