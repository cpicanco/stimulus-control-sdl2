unit Generics.Iterator.Table.Contract;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}

interface

type

  { ITableIterator }

  generic ITableIterator<_GT> = interface
    function GetCurrent : _GT;
    function IsCurrentEmpty(out ACell: _GT) : Boolean;
    function IsFirstRow: Boolean;
    function IsLastRow: Boolean;
    function IsFirstCol: Boolean;
    function IsLastCol: Boolean;
    procedure GoFirstRow;
    procedure GoNextRow;
    procedure GoPreviousRow;
    procedure GoLastRow;
    procedure GoFirstCol;
    procedure GoNextCol;
    procedure GoPreviousCol;
    procedure GoLastCol;
  end;

implementation

end.
