unit Generics.Map;

{$mode objfpc}{$H+}

interface

uses
  Generics.Collections;

type

  { TObjectToIntegerMap }

  generic TObjectToIntegerMap<T: class> = class
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

constructor TObjectToIntegerMap.Create;
begin
  inherited Create;
  FDictionary := TDictionaryType.Create;
end;

destructor TObjectToIntegerMap.Destroy;
begin
  FDictionary.Free;
  inherited Destroy;
end;

procedure TObjectToIntegerMap.Add(const Key: T; Value: Integer);
begin
  FDictionary.Add(Key, Value);
end;

procedure TObjectToIntegerMap.Remove(const Key: T);
begin
  FDictionary.Remove(Key);
end;

function TObjectToIntegerMap.ContainsKey(const Key: T): Boolean;
begin
  Result := FDictionary.ContainsKey(Key);
end;

function TObjectToIntegerMap.TryGetValue(const Key: T; out Value: Integer): Boolean;
begin
  Result := FDictionary.TryGetValue(Key, Value);
end;

function TObjectToIntegerMap.Sum: Integer;
var
  Pair: TDictionaryType.TDictionaryPair;
begin
  Result := 0;
  for Pair in FDictionary do
    Result := Result + Pair.Value;
end;

procedure TObjectToIntegerMap.Clear;
begin
  FDictionary.Clear;
end;

function TObjectToIntegerMap.GetItem(const Key: T): Integer;
begin
  Result := FDictionary[Key];
end;

procedure TObjectToIntegerMap.SetItem(const Key: T; const Value: Integer);
begin
  FDictionary[Key] := Value;
end;

end.
