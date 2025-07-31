{
  Stimulus Control
  Copyright (C) 2023-2025 Carlos Rafael Fernandes Picanço.

  The present file is distributed under the terms of the GNU General Public License (GPL v3.0).

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
}
unit session.parameters.types;

{$mode ObjFPC}{$H+}

{$INTERFACES CORBA}
{$MODESWITCH ADVANCEDRECORDS}
{$MODESWITCH TYPEHELPERS}

interface

uses
  Classes, SysUtils, Variants, Generics.Collections;

type

  TParameterData = record
    Data: Variant;
    ClassField: Pointer;
  end;

  { TParameterDataHelper }

  TParameterDataHelper = type helper for TParameterData
    function ToString : string;
    function DataToString : string;
    function FieldToString : string;
  end;

  TParameter = record
    Key : string;
    Value : TParameterData;
  end;

  { TParameterHelper }

  TParameterHelper = type helper for TParameter
    function ToString : string;
  end;


  TParametersBase = specialize TDictionary<string, TParameterData>;

  IParameterEnumerator = interface(IInterface)
    ['{7D96F7BE-37EE-4EF2-A9BB-5C68597C7D19}']
    function GetCurrent: TParameter;
    function MoveNext: Boolean;
    procedure Reset;
    property Current: TParameter read GetCurrent;
  end;

  IEnumerableParameters = interface
    ['{A6F4C5F2-6076-464B-BEB0-47C9939E71BF}']
    function GetEnumerator: IParameterEnumerator;
  end;

  { TParametersEnumerator }

  TParametersEnumerator = class(TInterfacedObject, IParameterEnumerator)
  private
    FParameters: TParametersBase;
    FIndex: Integer;
  public
    constructor Create(AParameters: TParametersBase);
    function GetCurrent: TParameter;
    function MoveNext: Boolean;
    procedure Reset;
    property Current: TParameter read GetCurrent;
  end;

  TParameters = class(TParametersBase, IEnumerableParameters)
    constructor Create; override;
    function GetEnumerator : IParameterEnumerator;
  end;

implementation

{ TParameterDataHelper }

function TParameterDataHelper.ToString: string;
begin
  Result := string.Join('=', [DataToString, FieldToString]);
end;

function TParameterDataHelper.DataToString: string;
begin
  Result := VarToStr(Self.Data);
end;

function TParameterDataHelper.FieldToString: string;
begin
  case VarType(Self.Data) of
    varSmallInt,
    varInteger,
    varShortInt,
    varByte,
    varWord,
    varLongWord,
    varInt64,
    varQWord : begin
      Result := IntToStr(Integer(Self.ClassField^));
    end;

    varBoolean: begin
      Result := BoolToStr(Boolean(Self.ClassField^), True);
    end;

    varString: begin
      Result := String(Self.ClassField^);
    end

    else
      raise EConvertError.Create('TVariantHelper.ToString');
  end;
end;

{ TParameterHelper }

function TParameterHelper.ToString: string;
begin
  Result := string.Join('=', [Self.Key, Self.Value.ToString]);
end;

{ TParametersEnumerator }

constructor TParametersEnumerator.Create(AParameters: TParametersBase);
begin
  FParameters := AParameters;
  FIndex := -1;
end;

function TParametersEnumerator.GetCurrent: TParameter;
begin
  Result.Key   := FParameters.Keys.ToArray[FIndex];
  Result.Value := FParameters.Values.ToArray[FIndex];
end;

function TParametersEnumerator.MoveNext: Boolean;
begin
  Inc(FIndex);
  Result := FIndex < FParameters.Count;
end;

procedure TParametersEnumerator.Reset;
begin
  FIndex := -1;
end;

constructor TParameters.Create;
begin
  inherited Create;

end;

function TParameters.GetEnumerator: IParameterEnumerator;
begin
  // TParametersEnumerator is reference counted
  Result := TParametersEnumerator.Create(Self) as IParameterEnumerator;
end;

end.

