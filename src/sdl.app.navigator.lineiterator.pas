unit sdl.app.navigator.lineiterator;

{$mode ObjFPC}{$H+}

interface

uses
  Classes, SysUtils, Generics.Collections,
  Generics.Aggregator,
  sdl.app.selectable.contract,
  sdl.app.selectable.list;

type

  { TPossibleSelections }

  TPossibleSelections = class (specialize TAggregator<ISelectable>)
  private
    FFirst : Boolean;
    FBaseControl : ISelectable;
  public
    function Next : ISelectable;
    function Previous : ISelectable;
    procedure Update(ASelectables : TSelectables);
    procedure SetBaseControl(ABaseControl : ISelectable);
  end;


implementation


{ TPossibleSelections }

function TPossibleSelections.Next: ISelectable;
begin
  if List.Count = 0 then begin
    Result := FBaseControl;
    if Result = nil then begin
      raise EArgumentNilException.Create('Base control = nil');
    end;
    Exit;
  end;

  if FFirst then begin
    Result := Iterator.GetCurrent;
    FFirst := False;
  end else begin
    with Iterator do begin
      if IsLast then begin
        GoFirst;
      end else begin
        GoNext;
      end;
      Result := GetCurrent;
    end;
  end;
end;

function TPossibleSelections.Previous: ISelectable;
begin
  if List.Count = 0 then begin
    Result := FBaseControl;
    if Result = nil then begin
      raise EArgumentNilException.Create('Base control = nil');
    end;
    Exit;
  end;

  if FFirst then begin
    Result := Iterator.GetCurrent;
    FFirst := False;
  end else begin
    with Iterator do begin
      if IsFirst then begin
        GoLast;
      end else begin
        GoPrevious;
      end;
      Result := GetCurrent;
    end;
  end;
end;

procedure TPossibleSelections.Update(ASelectables: TSelectables);
var
  LISelectable : ISelectable;
begin
  FFirst := True;
  List.Clear;
  if ASelectables = nil then Exit;
  for LISelectable in ASelectables do begin
    List.Add(LISelectable);
  end;
  Iterator.GoFirst;
end;

procedure TPossibleSelections.SetBaseControl(ABaseControl: ISelectable);
begin
  FBaseControl := ABaseControl;
end;

end.

