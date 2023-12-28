unit sdl.app.controller.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses
  Classes, SysUtils, sdl.app.navigator.contract;

type

  { IController }

  IController = interface
    ['{E249621B-0923-4048-A6FB-98CAF19CB6A2}']
    function Navigator : ITableNavigator;
  end;

implementation

end.

