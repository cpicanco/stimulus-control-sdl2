unit sdl.app.selectable.contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

uses SDL2;

type

  { ISelectable }

  ISelectable = interface
    ['{3914BD43-1105-4C45-BD28-9F1709AC16AB}']
    function Origen : TSDL_Point;
    procedure Unselect;
    procedure Select;
    procedure Confirm;
  end;

implementation

end.
