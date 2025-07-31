{
  Stimulus Control
  Copyright (C) 2024-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit sdl.app.controller.manager;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, SDL2,
  sdl.app.events.abstract,
  sdl.app.controller.contract,
  sdl.app.controller.types,
  sdl.app.controller;

type

  { TControllerManager }

  TControllerManager = class
    private
      FController : TController;
      FOnControllerDeviceAdded: TOnControllerDeviceAddedEvent;
      FOnControllerDeviceRemapped: TOnControllerDeviceRemappedEvent;
      FOnControllerDeviceRemoved: TOnControllerDeviceRemovedEvent;
      FOnJoyDeviceAdded: TOnJoyDeviceAddedEvent;
      FOnJoyDeviceRemoved: TOnJoyDeviceRemovedEvent;
      function GetFirstController: IController;
      procedure JoyDeviceAdded(const event: TSDL_JoyDeviceEvent);
      procedure JoyDeviceRemoved(const event: TSDL_JoyDeviceEvent);

      procedure ControllerDeviceAdded(
        const event: TSDL_ControllerDeviceEvent);
      procedure ControllerDeviceRemoved(
        const event: TSDL_ControllerDeviceEvent);
      procedure ControllerDeviceRemapped(
        const event: TSDL_ControllerDeviceEvent);
      procedure SetOnControllerDeviceAdded(
        AValue: TOnControllerDeviceAddedEvent);
      procedure SetOnControllerDeviceRemapped(
        AValue: TOnControllerDeviceRemappedEvent);
      procedure SetOnControllerDeviceRemoved(
        AValue: TOnControllerDeviceRemovedEvent);
      procedure SetOnJoyDeviceAdded(AValue: TOnJoyDeviceAddedEvent);
      procedure SetOnJoyDeviceRemoved(AValue: TOnJoyDeviceRemovedEvent);
    public
      constructor Create;
      destructor Destroy; override;
      procedure CreateController(AController : TControllerCode); overload;
      procedure CreateController(AController : integer); overload;
      procedure Disable;
      procedure Reboot;
      property FirstController : IController read GetFirstController;
      property OnJoyDeviceAdded : TOnJoyDeviceAddedEvent
        read FOnJoyDeviceAdded write SetOnJoyDeviceAdded;
      property OnJoyDeviceRemoved : TOnJoyDeviceRemovedEvent
        read FOnJoyDeviceRemoved write SetOnJoyDeviceRemoved;

      property OnControllerDeviceAdded : TOnControllerDeviceAddedEvent
        read FOnControllerDeviceAdded write SetOnControllerDeviceAdded;
      property OnControllerDeviceRemoved : TOnControllerDeviceRemovedEvent
        read FOnControllerDeviceRemoved write SetOnControllerDeviceRemoved;
      property OnControllerDeviceRemapped : TOnControllerDeviceRemappedEvent
        read FOnControllerDeviceRemapped write SetOnControllerDeviceRemapped;
  end;

var
  Controllers : TControllerManager;

implementation

uses sdl.app.controller.factory, sdl.app.trials.factory;

{ TControllerManager }

procedure TControllerManager.JoyDeviceAdded(const event: TSDL_JoyDeviceEvent);
begin

end;

procedure TControllerManager.JoyDeviceRemoved(const event: TSDL_JoyDeviceEvent);
begin

end;

procedure TControllerManager.ControllerDeviceAdded(
  const event: TSDL_ControllerDeviceEvent);
begin

end;

procedure TControllerManager.ControllerDeviceRemoved(
  const event: TSDL_ControllerDeviceEvent);
begin

end;

procedure TControllerManager.ControllerDeviceRemapped(
  const event: TSDL_ControllerDeviceEvent);
begin

end;

procedure TControllerManager.SetOnControllerDeviceAdded(
  AValue: TOnControllerDeviceAddedEvent);
begin
  if FOnControllerDeviceAdded = AValue then Exit;
  FOnControllerDeviceAdded := AValue;
end;

procedure TControllerManager.SetOnControllerDeviceRemapped(
  AValue: TOnControllerDeviceRemappedEvent);
begin
  if FOnControllerDeviceRemapped = AValue then Exit;
  FOnControllerDeviceRemapped := AValue;
end;

procedure TControllerManager.SetOnControllerDeviceRemoved(
  AValue: TOnControllerDeviceRemovedEvent);
begin
  if FOnControllerDeviceRemoved = AValue then Exit;
  FOnControllerDeviceRemoved := AValue;
end;

procedure TControllerManager.SetOnJoyDeviceAdded(
  AValue: TOnJoyDeviceAddedEvent);
begin
  if FOnJoyDeviceAdded = AValue then Exit;
  FOnJoyDeviceAdded := AValue;
end;

procedure TControllerManager.SetOnJoyDeviceRemoved(
  AValue: TOnJoyDeviceRemovedEvent);
begin
  if FOnJoyDeviceRemoved = AValue then Exit;
  FOnJoyDeviceRemoved := AValue;
end;

procedure TControllerManager.CreateController(
  AController: TControllerCode);
begin
  FController := TControllerFactory.New(AController);
end;

procedure TControllerManager.CreateController(AController: integer);
begin
  CreateController(TControllerCode(AController));
end;

procedure TControllerManager.Disable;
begin
  FController.Free;
  FController := nil;
end;

procedure TControllerManager.Reboot;
var
  LControllerCode : TControllerCode;
begin
  if Assigned(FController) then begin
    LControllerCode := FController.ControllerCode;
    Disable;
    CreateController(LControllerCode);
    TTrialFactory.UpdateNavigator;
  end;
end;

constructor TControllerManager.Create;
begin

end;

destructor TControllerManager.Destroy;
begin
  if Assigned(FController) then begin
    FController.Free;
  end;
  inherited Destroy;
end;

function TControllerManager.GetFirstController: IController;
begin
  Result := FController as IController;
end;

end.

