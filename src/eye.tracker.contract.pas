{
  Stimulus Control
  Copyright (C) 2014-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit eye.tracker.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses
  Classes, eye.tracker.types;

type

  { IEyeTracker }

  IEyeTracker = interface
    ['{510AF7AB-0F7F-416A-8B97-4CBE13950CB4}']
    function UseGazeAsInput : Boolean;
    function GetGazeOnScreenEvent : TGazeOnScreenEvent;
    function CurrentGazes : TNormalizedGazes;
    function TrackerClassName : string;
    function IsFake : Boolean;
    procedure SetGazeOnScreenEvent(AValue: TGazeOnScreenEvent);
    procedure SetOnCalibrationSuccessful(AValue: TNotifyEvent);
    procedure SetOnCalibrationFailed(AValue: TNotifyEvent);
    procedure SetDataFilename(AFilename : string);
    procedure StartRecording;
    procedure StopRecording;
    procedure StartCalibration;
    procedure StopCalibration;
    procedure CalibrationSuccessful;
  end;

implementation

end.

