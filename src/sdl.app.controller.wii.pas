{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.controller.wii;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, ctypes, SDL2,
  sdl.app.events.abstract,
  sdl.app.controller,
  sdl.app.controller.axis;

type

  // it not so simple to use wii remote, current implementation does not work

(*
  In Windows 11 22H2, the old Devices and Printers interface can still be
  accessed by running the following command from the Run dialog (Windows Key+R):

  shell:::{A8A91A66-3A7D-4424-8D24-04E180695C7A}

  1) Click on Add Device.
  2) Press the red button inside or press button 1 and 2 together.
  3) Select Nintendo-RVL-CNT-01
  4) A PIN will be requested. Click next with empty PIN.
  5) Wait for installation completion.
*)

  { TSDLWiiRemote }

  TSDLWiiRemote = class(TController)
  private
    FWidth  : LongInt;
    FHeight : LongInt;
    FGameController : PSDL_GameController;
    F2DAxis : TController2DAxis;
    FOnControllerAxisMotion: TOnControllerAxisMotionEvent;
    FOnControllerButtonDown: TOnControllerButtonDownEvent;
    FOnControllerButtonUp: TOnControllerButtonUpEvent;
    procedure ControllerAxisMotion(const event: TSDL_ControllerAxisEvent);
    procedure ControllerButtonDown(const event: TSDL_ControllerButtonEvent);
    procedure ControllerButtonUp(const event: TSDL_ControllerButtonEvent);
    procedure ControllerTouchPadMotion(const event: TSDL_ControllerTouchpadEvent);
    procedure ControllerSensorUpdate(const event: TSDL_ControllerSensorEvent);
    procedure SetOnControllerAxisMotion(
      AValue: TOnControllerAxisMotionEvent);
    procedure SetOnControllerButtonDown(
      AValue: TOnControllerButtonDownEvent);
    procedure SetOnControllerButtonUp(
      AValue: TOnControllerButtonUpEvent);
  public
    constructor Create; override;
    destructor Destroy; override;
    property OnControllerAxisMotion : TOnControllerAxisMotionEvent
      read FOnControllerAxisMotion write SetOnControllerAxisMotion;

    property OnControllerButtonDown : TOnControllerButtonDownEvent
      read FOnControllerButtonDown write SetOnControllerButtonDown;

    property OnControllerButtonUp : TOnControllerButtonUpEvent
      read FOnControllerButtonUp write SetOnControllerButtonUp;
  end;

implementation

uses
  Math,
  //sdl.app.output,
  sdl.app.events.custom,
  sdl.app.video.methods,
  sdl.app.controller.types,
  sdl.app.graphics.debug,
  timestamps;

var
  LastClock : Float = 0;

{ TSDLWiiRemote }

procedure TSDLWiiRemote.ControllerAxisMotion(
  const event: TSDL_ControllerAxisEvent);
begin
  if (ClockMonotonic-LastClock) < 0.250 then Exit;

  if F2DAxis.IsOutSideDeadZone then begin
    LastClock := ClockMonotonic;
    //DrawDebugCircle(LDegrees);
    case F2DAxis.CurrentDirection of
      Top:
        FNavigator.GoTop;
      Left:
        FNavigator.GoLeft;
      Bottom:
        FNavigator.GoBottom;
      Right:
        FNavigator.GoRight;
      TopLeft:
        FNavigator.GoTopLeft;
      TopRight:
        FNavigator.GoTopRight;
      BottomLeft:
        FNavigator.GoBottomLeft;
      BottomRight:
        FNavigator.GoBottomRight;
      None: { do nothing };
    end;
  end;
end;

procedure TSDLWiiRemote.ControllerButtonDown(
  const event: TSDL_ControllerButtonEvent);
begin

end;

procedure TSDLWiiRemote.ControllerButtonUp(
  const event: TSDL_ControllerButtonEvent);
begin

end;

procedure TSDLWiiRemote.ControllerTouchPadMotion(
  const event: TSDL_ControllerTouchpadEvent);
begin

end;

procedure TSDLWiiRemote.ControllerSensorUpdate(
  const event: TSDL_ControllerSensorEvent);
begin

end;

procedure TSDLWiiRemote.SetOnControllerAxisMotion(
  AValue: TOnControllerAxisMotionEvent);
begin
  if FOnControllerAxisMotion = AValue then Exit;
  FOnControllerAxisMotion := AValue;
end;

procedure TSDLWiiRemote.SetOnControllerButtonDown(
  AValue: TOnControllerButtonDownEvent);
begin
  if FOnControllerButtonDown = AValue then Exit;
  FOnControllerButtonDown := AValue;
end;

procedure TSDLWiiRemote.SetOnControllerButtonUp(
  AValue: TOnControllerButtonUpEvent);
begin
  if FOnControllerButtonUp = AValue then Exit;
  FOnControllerButtonUp := AValue;
end;

constructor TSDLWiiRemote.Create;
var
  LWindowRect: TSDL_Rect;
  LGameControllers: cint;
begin
  LWindowRect := WindowBoundsRect;
  FWidth := LWindowRect.w;
  FHeight := LWindowRect.h;
  F2DAxis := TController2DAxis.Create;

  SDLEvents.OnControllerButtonDown := @ControllerButtonDown;
  SDLEvents.OnControllerAxisMotion := @ControllerAxisMotion;
  SDLEvents.OnControllerButtonUp := @ControllerButtonUp;
  SDLEvents.OnControllerSensorUpdate := @ControllerSensorUpdate;
  SDLEvents.OnControllerTouchPadMotion := @ControllerTouchPadMotion;

  SDL_JoystickUpdate;
  LGameControllers := SDL_NumJoysticks;
  //Print('NumJoysticks:'+LGameControllers.ToString);
  if LGameControllers > 0 then begin
    // '?'
    //Print(SDL_GameControllerNameForIndex(0));
    FGameController := SDL_GameControllerOpen(0);
    if FGameController <> nil then begin
      F2DAxis.GameController := FGameController;
      if SDL_GameControllerHasRumble(FGameController) = SDL_TRUE then begin
        //Print('Has Rumble');
      end;

      if SDL_GameControllerHasRumbleTriggers(FGameController) = SDL_TRUE then begin
        //Print('Has Rumble Triggers');
      end;

      if SDL_GameControllerHasSensor(FGameController, SDL_SENSOR_ACCEL) = SDL_TRUE then begin
        //Print('Has Accelerometer');
      end;

      if SDL_GameControllerHasSensor(FGameController, SDL_SENSOR_GYRO) = SDL_TRUE then begin
        //Print('Has Gyroscope');
      end;

    end else begin
      raise EArgumentNilException.Create('Controller Error.');
    end;
  end else begin
    raise EArgumentNilException.Create('No controller found.');
  end;
end;

destructor TSDLWiiRemote.Destroy;
begin
  F2DAxis.Free;
  inherited Destroy;
end;

end.


