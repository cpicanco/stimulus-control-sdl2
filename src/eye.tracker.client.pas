{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit eye.tracker.client;

{$mode ObjFPC}{$H+}

interface

uses Classes, eye.tracker.contract, eye.tracker.types;

type

  { TEyeTrackerClient }

  TEyeTrackerClient = class(TObject, IEyeTracker)
    protected
      function UseGazeAsInput : Boolean;
      function GetGazeOnScreenEvent : TGazeOnScreenEvent; virtual; abstract;
      function IsFake : Boolean; virtual; abstract;
      function TrackerClassName : string; virtual; abstract;
      procedure SetGazeOnScreenEvent(AValue: TGazeOnScreenEvent); virtual; abstract;
      procedure SetOnCalibrationSuccessful(AValue: TNotifyEvent); virtual; abstract;
      procedure SetOnCalibrationFailed(AValue: TNotifyEvent); virtual; abstract;
      procedure SetDataFilename(AFilename : string); virtual; abstract;
      procedure StartRecording; virtual; abstract;
      procedure StopRecording; virtual; abstract;
      procedure StartCalibration; virtual; abstract;
      procedure StopCalibration; virtual; abstract;
      procedure CalibrationSuccessful; virtual; abstract;
    public
      class function Exists : Boolean;
      function CurrentGazes : TNormalizedGazes; virtual; abstract;
  end;

implementation

uses eye.tracker, session.parameters.global;

{ TEyeTrackerClient }

function TEyeTrackerClient.UseGazeAsInput: Boolean;
begin
  Result := GlobalTrialParameters.UseGazeAsInput;
end;

class function TEyeTrackerClient.Exists: Boolean;
begin
  Result := EyeTracker <> nil;
end;

end.

