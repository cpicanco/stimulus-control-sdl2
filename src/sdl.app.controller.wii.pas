unit sdl.app.controller.wii;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, SDL2,
  sdl.app.events.abstract,
  sdl.app.controller;

type

  { TSDLWiiRemote }

  TSDLWiiRemote = class(TController)
  private
    FOnControllerAxisMotion: TOnControllerAxisMotionEvent;
    FOnControllerButtonDown: TOnControllerButtonDownEvent;
    FOnControllerButtonUp: TOnControllerButtonUpEvent;
    procedure ControllerAxisMotionEvent(
      const event: TSDL_ControllerAxisEvent);
    procedure ControllerButtonDownEvent(
      const event: TSDL_ControllerButtonEvent);
    procedure ControllerButtonUpEvent(
      const event: TSDL_ControllerButtonEvent);
    procedure ControllerDeviceAddedEvent(
      const event: TSDL_ControllerDeviceEvent);
    procedure ControllerDeviceRemovedEvent(
      const event: TSDL_ControllerDeviceEvent);
    procedure SetOnControllerAxisMotion(
      AValue: TOnControllerAxisMotionEvent);
    procedure SetOnControllerButtonDown(
      AValue: TOnControllerButtonDownEvent);
    procedure SetOnControllerButtonUp(
      AValue: TOnControllerButtonUpEvent);
  public
    property OnControllerAxisMotion : TOnControllerAxisMotionEvent
      read FOnControllerAxisMotion write SetOnControllerAxisMotion;

    property OnControllerButtonDown : TOnControllerButtonDownEvent
      read FOnControllerButtonDown write SetOnControllerButtonDown;

    property OnControllerButtonUp : TOnControllerButtonUpEvent
      read FOnControllerButtonUp write SetOnControllerButtonUp;
  end;

implementation

{ TSDLWiiRemote }

procedure TSDLWiiRemote.ControllerAxisMotionEvent(
  const event: TSDL_ControllerAxisEvent);
begin

end;

procedure TSDLWiiRemote.ControllerButtonDownEvent(
  const event: TSDL_ControllerButtonEvent);
begin

end;

procedure TSDLWiiRemote.ControllerButtonUpEvent(
  const event: TSDL_ControllerButtonEvent);
begin

end;

procedure TSDLWiiRemote.ControllerDeviceAddedEvent(
  const event: TSDL_ControllerDeviceEvent);
begin

end;

procedure TSDLWiiRemote.ControllerDeviceRemovedEvent(
  const event: TSDL_ControllerDeviceEvent);
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

end.


