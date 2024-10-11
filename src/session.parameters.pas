unit session.parameters;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Variants
  , session.parameters.types;

type
  { TParametricObject }

  TParametricObject = class
  private
    FOnAfterLoadingParameters: TNotifyEvent;
    FOnBeforeLoadingParameters: TNotifyEvent;
    FParameters: TParameters;
    procedure SetOnAfterLoadingParameters(AValue: TNotifyEvent);
    procedure SetOnBeforeLoadingParameters(AValue: TNotifyEvent);
  protected
    function GetValue(const AKey: string): string;
    procedure SetValue(const AKey: string; AValue: string);
    procedure RegisterParameter(const AKey: string;
      AClassField: Pointer; ADefaultValue: Variant);
  public
    constructor Create;
    destructor Destroy; override;
    procedure LoadParameters(AParameters : TStringList);
    procedure AssignParameters(AParameters : TStringList); virtual;
    property Values[const AKey: string]: string
      read GetValue write SetValue;
    property Parameters : TParameters read FParameters;
    property OnAfterLoadingParameters : TNotifyEvent
      read FOnAfterLoadingParameters write SetOnAfterLoadingParameters;
    property OnBeforeLoadingParameters : TNotifyEvent
      read FOnBeforeLoadingParameters write SetOnBeforeLoadingParameters;
  end;

implementation

{ TParametricObject }

constructor TParametricObject.Create;
begin
  inherited Create;
  FParameters := TParameters.Create;
end;

destructor TParametricObject.Destroy;
begin
  FParameters.Free;
  inherited Destroy;
end;

procedure TParametricObject.LoadParameters(AParameters: TStringList);
var
  LName : string;
  LValue : string;
  i: Integer;
begin
  if Assigned(OnBeforeLoadingParameters) then begin
    OnBeforeLoadingParameters(AParameters);
  end;
  for i := 0 to AParameters.Count -1 do begin
    AParameters.GetNameValue(i, LName, LValue);
    if FParameters.ContainsKey(LName) then begin
      Values[LName] := LValue;
    end;
  end;
  if Assigned(OnAfterLoadingParameters) then begin
    OnAfterLoadingParameters(Self);
  end;
end;

procedure TParametricObject.AssignParameters(AParameters: TStringList);
var
  LParameter: TParameter;
begin
  for LParameter in FParameters do
    AParameters.Values[LParameter.Key] := Values[LParameter.Key];
end;

procedure TParametricObject.RegisterParameter(const AKey: string;
  AClassField: Pointer; ADefaultValue: Variant);
var
  LParameter: TParameterData;
begin
  LParameter.Data := ADefaultValue;
  LParameter.ClassField := AClassField;
  FParameters.Add(AKey, LParameter);
end;

procedure TParametricObject.SetOnAfterLoadingParameters(AValue: TNotifyEvent);
begin
  if FOnAfterLoadingParameters = AValue then Exit;
  FOnAfterLoadingParameters := AValue;
end;

procedure TParametricObject.SetOnBeforeLoadingParameters(AValue: TNotifyEvent);
begin
  if FOnBeforeLoadingParameters = AValue then Exit;
  FOnBeforeLoadingParameters := AValue;
end;

function TParametricObject.GetValue(const AKey: string): string;
begin
  Result := FParameters[AKey].FieldToString;
end;

procedure TParametricObject.SetValue(const AKey: string; AValue: string);
var
  LParameter : TParameterData;
begin
  LParameter := FParameters[AKey];
  case VarType(LParameter.Data) of
    varSmallint: begin
      LParameter.Data :=
        StrToIntDef(AValue, Integer(FParameters[AKey].Data));
      SmallInt(FParameters[AKey].ClassField^) := LParameter.Data;
    end;

    varInteger : begin
      LParameter.Data :=
        StrToIntDef(AValue, Integer(FParameters[AKey].Data));
      Integer(FParameters[AKey].ClassField^) := LParameter.Data;
    end;

    varShortInt: begin
      LParameter.Data :=
        StrToIntDef(AValue, Integer(FParameters[AKey].Data));
      ShortInt(FParameters[AKey].ClassField^) := LParameter.Data;
    end;

    varByte: begin
      LParameter.Data :=
        StrToIntDef(AValue, Integer(FParameters[AKey].Data));
      Byte(FParameters[AKey].ClassField^) := LParameter.Data;
    end;

    varWord: begin
      LParameter.Data :=
        StrToIntDef(AValue, Integer(FParameters[AKey].Data));
      Word(FParameters[AKey].ClassField^) := LParameter.Data;
    end;

    varLongWord: begin
      LParameter.Data :=
        StrToIntDef(AValue, Integer(FParameters[AKey].Data));
      LongWord(FParameters[AKey].ClassField^) := LParameter.Data;
    end;

    varInt64: begin
      LParameter.Data :=
        StrToIntDef(AValue, Integer(FParameters[AKey].Data));
      Int64(FParameters[AKey].ClassField^) := LParameter.Data;
    end;

    varQWord: begin
      LParameter.Data :=
        StrToIntDef(AValue, Integer(FParameters[AKey].Data));
      QWord(FParameters[AKey].ClassField^) := LParameter.Data;
    end;

    varBoolean: begin
      LParameter.Data :=
        StrToBoolDef(AValue, Boolean(FParameters[AKey].Data));
      Boolean(FParameters[AKey].ClassField^) := LParameter.Data;
    end;

    varString: begin
      LParameter.Data := AValue;
      String(FParameters[AKey].ClassField^) := LParameter.Data;
    end;
    else
      raise EConvertError.Create(ClassName+'.SetValue:'+AKey);
  end;
  FParameters[AKey] := LParameter;
end;

end.
