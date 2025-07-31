unit sdl.app.controller.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses
  sdl.app.navigator.contract;

type

  { IController }

  IController = interface
    ['{E249621B-0923-2025-A6FB-98CAF19CB6A2}']
    function Navigator : ITableNavigator;
    procedure Hide;
    procedure Show;
  end;

implementation

end.

