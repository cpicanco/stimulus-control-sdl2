unit sdl.app.controller;

{$mode ObjFPC}{$H+}

interface

uses Classes, SysUtils,
  sdl.app.controller.contract,
  sdl.app.navigator.contract,
  sdl.app.navigator;

type

  { TController }

  TController = class(IController)
    protected
      FNavigator : TTableNavigator;
      function Navigator : ITableNavigator;
    public
      constructor Create; virtual;
      destructor Destroy; override;
  end;

implementation

{ TController }

function TController.Navigator: ITableNavigator;
begin
  Result := FNavigator as ITableNavigator;
end;

constructor TController.Create;
begin
  FNavigator := TTableNavigator.Create;
end;

destructor TController.Destroy;
begin
  FNavigator.Free;
end;

end.

