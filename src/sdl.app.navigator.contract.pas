unit sdl.app.navigator.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses sdl.app.selectable.contract, sdl.app.selectable.list;

type

  { ILineNavigator }

  ILineNavigator = interface
    ['{DEBD72A2-BC97-4D1C-80AB-A0CAB06BAEFC}']
    procedure Unselect;
    procedure SelectNext;
    procedure SelectPrevious;
    procedure ConfirmSelection;
    procedure SetBaseControl(AControl : ISelectable);
    procedure UpdateNavigationControls(AControls : TSelectables);
  end;

  ITableNavigator = interface
    ['{DEBD72A2-BC97-4D1C-80AB-A0CAB06BAEFC}']
    procedure Unselect;
    procedure SelectUp;
    procedure SelectDown;
    procedure SelectLeft;
    procedure SelectRight;
    //procedure GoLevelUp;
    //procedure GoLevelDown;
    procedure ConfirmSelection;
    procedure SetBaseControl(AControl : ISelectable);
    procedure UpdateNavigationControls(AControls : TSelectables);
  end;


implementation

end.
