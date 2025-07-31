unit Generics.Map;

{$mode objfpc}{$H+}

interface

uses
  Generics.Collections;

type

  { TObjectToIntegerMap }

  generic TGenericToIntegerMap<T: class> = class
  private
    type
      TDictionaryType = specialize TDictionary<T, Integer>;
    var
      FDictionary: TDictionaryType;
    function GetItem(const Key: T): Integer;
    procedure SetItem(const Key: T; const Value: Integer);
  public
    constructor Create;
    destructor Destroy; override;

    procedure Add(const Key: T; Value: Integer);
    procedure Remove(const Key: T);
    function ContainsKey(const Key: T): Boolean;
    function TryGetValue(const Key: T; out Value: Integer): Boolean;
    function Sum : integer;
    procedure Clear;

    property Items[const Key: T]: Integer read GetItem write SetItem; default;
  end;

implementation

constructor TGenericToIntegerMap.Create;
begin
  inherited Create;
  FDictionary := TDictionaryType.Create;
end;

destructor TGenericToIntegerMap.Destroy;
begin
  FDictionary.Free;
  inherited Destroy;
end;

procedure TGenericToIntegerMap.Add(const Key: T; Value: Integer);
begin
  FDictionary.Add(Key, Value);
end;

procedure TGenericToIntegerMap.Remove(const Key: T);
begin
  FDictionary.Remove(Key);
end;

function TGenericToIntegerMap.ContainsKey(const Key: T): Boolean;
begin
  Result := FDictionary.ContainsKey(Key);
end;

function TGenericToIntegerMap.TryGetValue(const Key: T; out Value: Integer): Boolean;
begin
  Result := FDictionary.TryGetValue(Key, Value);
end;

function TGenericToIntegerMap.Sum: Integer;
var
  Pair: TDictionaryType.TDictionaryPair;
begin
  Result := 0;
  for Pair in FDictionary do
    Result := Result + Pair.Value;
end;

procedure TGenericToIntegerMap.Clear;
begin
  FDictionary.Clear;
end;

function TGenericToIntegerMap.GetItem(const Key: T): Integer;
begin
  Result := FDictionary[Key];
end;

procedure TGenericToIntegerMap.SetItem(const Key: T; const Value: Integer);
begin
  FDictionary[Key] := Value;
end;

end.
