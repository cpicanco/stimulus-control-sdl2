unit sdl.app.navigator;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils,
  sdl.app.navigator.contract,
  sdl.app.selectable.contract,
  sdl.app.selectable.list,
  sdl.app.navigator.tableiterator;

type

  { TTableNavigator }

  TTableNavigator = class(ITableNavigator)
  private
    FControls : TPossibleSelections;
    FCurrentControl : ISelectable;
    procedure SetBaseControl(ABaseControl : ISelectable);
    procedure UpdateNavigationControls(AControls: TSelectables);
  public
    constructor Create;
    destructor Destroy; override;
    procedure Unselect;
    procedure Select;
    procedure SelectUp;
    procedure SelectDown;
    procedure SelectLeft;
    procedure SelectRight;
    procedure ConfirmSelection;
  end;

implementation

{ TTableNavigator }

procedure TTableNavigator.UpdateNavigationControls(
  AControls: TSelectables);
begin
  FControls.Update(AControls);
end;

procedure TTableNavigator.SetBaseControl(ABaseControl: ISelectable);
begin
  FControls.SetBaseControl(ABaseControl);
end;

constructor TTableNavigator.Create;
begin
  inherited Create;
  FControls := TPossibleSelections.Create;
end;

destructor TTableNavigator.Destroy;
begin
  FControls.Free;
end;

procedure TTableNavigator.Unselect;
begin
  FCurrentControl.Unselect;
end;

procedure TTableNavigator.Select;
begin
  FCurrentControl := FControls.Select;
  FCurrentControl.Select;
end;

procedure TTableNavigator.SelectUp;
begin
  FCurrentControl := FControls.PreviousRow;
  FCurrentControl.Select;
end;

procedure TTableNavigator.SelectDown;
begin
  FCurrentControl := FControls.NextRow;
  FCurrentControl.Select;
end;

procedure TTableNavigator.SelectLeft;
begin
  FCurrentControl := FControls.PreviousCol;
  FCurrentControl.Select;
end;

procedure TTableNavigator.SelectRight;
begin
  FCurrentControl := FControls.NextCol;
  FCurrentControl.Select;
end;

procedure TTableNavigator.ConfirmSelection;
begin
  FCurrentControl.Confirm;
end;

end.

