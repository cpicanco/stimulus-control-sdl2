unit sdl.app.navigable.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses sdl.app.navigator.contract;

type

  INavigable = interface
  ['{8919BF2E-EE00-4873-ACF0-40222299A426}']
    procedure UpdateNavigator;
    procedure SetNavigator(ANavigator : ITableNavigator);
  end;

implementation

end.

