{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.controller.ps4;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, ctypes, SDL2,
  sdl.app.events.abstract,
  sdl.app.controller,
  sdl.app.controller.axis;

type

  { TSDLPS4Controller }

  TSDLPS4Controller = class(TController)
  private
    FWidth : integer;
    FHeight: integer;
    FGameController : PSDL_GameController;
    F2DAxis : TController2DAxis;
    FOnControllerAxisMotion: TOnControllerAxisMotionEvent;
    FOnControllerButtonDown: TOnControllerButtonDownEvent;
    FOnControllerButtonUp: TOnControllerButtonUpEvent;
    FOnControllerTouchPadMotion: TOnControllerTouchPadMotionEvent;
    FOnControllerSensorUpdate: TOnControllerSensorUpdateEvent;
    procedure ControllerAxisMotion(const event: TSDL_ControllerAxisEvent);
    procedure ControllerButtonDown(const event: TSDL_ControllerButtonEvent);
    procedure ControllerButtonUp(const event: TSDL_ControllerButtonEvent);
    procedure ControllerTouchPadMotion(const event: TSDL_ControllerTouchpadEvent);
    procedure ControllerSensorUpdate(const event: TSDL_ControllerSensorEvent);
    procedure SetOnControllerAxisMotion(AValue: TOnControllerAxisMotionEvent);
    procedure SetOnControllerButtonDown(AValue: TOnControllerButtonDownEvent);
    procedure SetOnControllerButtonUp(AValue: TOnControllerButtonUpEvent);
    procedure SetOnControllerTouchPadMotion(
      AValue: TOnControllerTouchPadMotionEvent);
    procedure SetOnControllerSensorUpdate(AValue: TOnControllerSensorUpdateEvent);
  public
    constructor Create; override;
    destructor Destroy; override;
    property OnControllerAxisMotion : TOnControllerAxisMotionEvent
      read FOnControllerAxisMotion write SetOnControllerAxisMotion;

    property OnControllerButtonDown : TOnControllerButtonDownEvent
      read FOnControllerButtonDown write SetOnControllerButtonDown;

    property OnControllerButtonUp : TOnControllerButtonUpEvent
      read FOnControllerButtonUp write SetOnControllerButtonUp;

    property OnControllerTouchPadMotion : TOnControllerTouchPadMotionEvent
      read FOnControllerTouchPadMotion write SetOnControllerTouchPadMotion;

    property OnControllerSensorUpdate : TOnControllerSensorUpdateEvent
      read FOnControllerSensorUpdate write SetOnControllerSensorUpdate;
  end;

implementation

uses
  Math,
  //sdl.app.output,
  sdl.app.events.custom,
  sdl.app.video.methods,
  sdl.app.controller.types;

{ TSDLPS4Controller }

const
  PS4_BUTTON_SHARE = 4;
  PS4_BUTTON_HOME = 5;
  PS4_BUTTON_OPTIONS = 6;
  //LHumbleTime = 100;

procedure TSDLPS4Controller.ControllerAxisMotion(
  const event: TSDL_ControllerAxisEvent);
begin
  with F2DAxis do begin
    if IsOutSideDeadZone then begin
      UpdateCurrentDirection;
      if CurrentDirection <> LastDirection then begin
        case CurrentDirection of
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
      LastDirection := CurrentDirection;

    end else begin
      LastDirection := None;
    end;
  end;
end;

procedure TSDLPS4Controller.ControllerButtonDown(
  const event: TSDL_ControllerButtonEvent);
begin
  //Print('');
  //Print('Button Event');
  //Print(event.button.ToString);
  //Print(IntToStr(event.which));
end;

procedure TSDLPS4Controller.ControllerButtonUp(
  const event: TSDL_ControllerButtonEvent);
begin
  case event.button of
    SDL_CONTROLLER_BUTTON_A: ;
    SDL_CONTROLLER_BUTTON_B: ;
    SDL_CONTROLLER_BUTTON_X: ;
    SDL_CONTROLLER_BUTTON_Y: ;

    PS4_BUTTON_SHARE: ;
    PS4_BUTTON_HOME: ;
    PS4_BUTTON_OPTIONS: ;

    SDL_CONTROLLER_BUTTON_LEFTSTICK: ;
    SDL_CONTROLLER_BUTTON_RIGHTSTICK: ;
    SDL_CONTROLLER_BUTTON_LEFTSHOULDER: FNavigator.ConfirmSelection;
    SDL_CONTROLLER_BUTTON_RIGHTSHOULDER: FNavigator.ConfirmSelection;

    SDL_CONTROLLER_BUTTON_DPAD_UP: begin
      FNavigator.GoTop;
    end;

    SDL_CONTROLLER_BUTTON_DPAD_DOWN: begin
      FNavigator.GoBottom;
    end;

    SDL_CONTROLLER_BUTTON_DPAD_LEFT: begin
      FNavigator.GoLeft;
    end;

    SDL_CONTROLLER_BUTTON_DPAD_RIGHT: begin
      FNavigator.GoRight;
    end;

    SDL_CONTROLLER_BUTTON_TOUCHPAD: ;
  end;
end;

procedure TSDLPS4Controller.ControllerTouchPadMotion(
  const event: TSDL_ControllerTouchpadEvent);
var
  touchpad, finger: cint32;
  state: Uint8;
  x, y, pressure: Float;
begin
  // Get the touchpad and finger indices
  touchpad := event.touchpad;
  finger := event.finger;

  // Get the finger state, position, and pressure
  SDL_GameControllerGetTouchpadFinger(
    FGameController, touchpad, finger, @state, @x, @y, @pressure);

  // Print the finger information
  //Print(Format('Touchpad %d, Finger %d, State %d, X %f, Y %f, Pressure %f',
  //  [touchpad, finger, state, x, y, pressure]));
end;

procedure TSDLPS4Controller.ControllerSensorUpdate(const event: TSDL_ControllerSensorEvent);
var
  LSensor: PSDL_Sensor;
begin
  LSensor := SDL_SensorFromInstanceID(event.which);
  if not Assigned(LSensor) then
  begin
    //Print('Couldn''t get sensor for sensor event');
    Exit;
  end;

  case SDL_SensorGetType(LSensor) of
    SDL_SENSOR_ACCEL: begin
      //Print(Format('Accelerometer update: %.2f, %.2f, %.2f',
      //  [event.data[0], event.data[1], event.data[2]]));
    end;

    SDL_SENSOR_GYRO: begin
      //Print(Format('Gyro update: %.2f, %.2f, %.2f',
      //  [event.data[0], event.data[1], event.data[2]]));
    end

    otherwise begin
      //Print(Format('Sensor update for sensor type %1d',
      //  [SDL_SensorGetType(LSensor)]));
    end;
  end;
end;

procedure TSDLPS4Controller.SetOnControllerAxisMotion(
  AValue: TOnControllerAxisMotionEvent);
begin
  if FOnControllerAxisMotion = AValue then Exit;
  FOnControllerAxisMotion := AValue;
end;

procedure TSDLPS4Controller.SetOnControllerButtonDown(
  AValue: TOnControllerButtonDownEvent);
begin
  if FOnControllerButtonDown = AValue then Exit;
  FOnControllerButtonDown := AValue;
end;

procedure TSDLPS4Controller.SetOnControllerButtonUp(
  AValue: TOnControllerButtonUpEvent);
begin
  if FOnControllerButtonUp = AValue then Exit;
  FOnControllerButtonUp := AValue;
end;

procedure TSDLPS4Controller.SetOnControllerTouchPadMotion(
  AValue: TOnControllerTouchPadMotionEvent);
begin
  if FOnControllerTouchPadMotion = AValue then Exit;
  FOnControllerTouchPadMotion := AValue;
end;

procedure TSDLPS4Controller.SetOnControllerSensorUpdate(
  AValue: TOnControllerSensorUpdateEvent);
begin
  if FOnControllerSensorUpdate = AValue then Exit;
  FOnControllerSensorUpdate := AValue;
end;

constructor TSDLPS4Controller.Create;
var
  LGameControllers : integer;
  LWindowRect : TSDL_Rect;
begin
  inherited Create;
  F2DAxis := TController2DAxis.Create;
  LWindowRect := WindowBoundsRect;
  FWidth := LWindowRect.w;
  FHeight := LWindowRect.h;

  SDLEvents.OnControllerButtonDown := @ControllerButtonDown;
  SDLEvents.OnControllerAxisMotion := @ControllerAxisMotion;
  SDLEvents.OnControllerButtonUp := @ControllerButtonUp;
  SDLEvents.OnControllerSensorUpdate := @ControllerSensorUpdate;
  SDLEvents.OnControllerTouchPadMotion := @ControllerTouchPadMotion;

  SDL_JoystickUpdate;
  LGameControllers := SDL_NumJoysticks;
  //Print('NumJoysticks:'+LGameControllers.ToString);
  if LGameControllers > 0 then begin
    // 'PS4 Controller'
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

destructor TSDLPS4Controller.Destroy;
begin
  F2DAxis.Free;
  SDLEvents.OnControllerButtonDown := nil;
  SDLEvents.OnControllerAxisMotion := nil;
  SDLEvents.OnControllerButtonUp := nil;
  SDLEvents.OnControllerSensorUpdate := nil;
  SDLEvents.OnControllerTouchPadMotion := nil;
  SDL_GameControllerClose(FGameController);
  inherited Destroy;
end;

end.


